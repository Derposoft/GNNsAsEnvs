from ray.rllib.agents import ppo
import numpy as np
import json
import os

# our code
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
from attention_study.model.attention_policy import PolicyModel
from attention_study.generate_baseline_metrics import parse_arguments, create_env_config, create_trainer_config
from attention_study.model.utils import load_edge_dictionary

# 3rd party
from attention_routing.nets.attention_model import AttentionModel
from attention_routing.problems.tsp.problem_tsp import TSP
from attention_routing.train import train_batch

# train artifact initialization imports
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from attention_routing.nets.critic_network import CriticNetwork
from attention_routing.options import get_options
from attention_routing.train import train_epoch, validate, get_inner_model
from attention_routing.reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from attention_routing.nets.attention_model import AttentionModel
from attention_routing.nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from attention_routing.utils import torch_load_cpu, load_problem

def initialize_train_artifacts(opts):
    '''
    code mostly from attention_routing/run.py:run(opts)
    repurposed for reinforcement learning here.
    :params None
    :returns optimizer, baseline, lr_scheduler, val_dataset, problem, tb_logger, opts
    '''
    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1
    
    return model, optimizer, baseline, lr_scheduler, tb_logger



TEST_SETTINGS = {
    'is_standalone': True, # are we training it in rllib, or standalone?
}

if __name__ == "__main__":
    # create training environment
    print('creating config')
    parser = parse_arguments()
    config = parser.parse_args()
    outer_configs, n_episodes = create_env_config(config)
    
    # 
    if TEST_SETTINGS['is_standalone']:
        # create model and training artifacts
        opts = get_options()
        model, optimizer, baseline, lr_scheduler, tb_logger =\
            initialize_train_artifacts(opts)
        
        # create model environment
        training_env = Figure8SquadRLLib(outer_configs)
        acs_edges_dict = load_edge_dictionary(training_env.map.g_acs.adj)
        
        # train using num_training_episodes episodes of episode_length length
        print('training')
        episode_length = 40
        num_training_episodes = 100
        for episode in range(num_training_episodes):
            agent_node = 0
            obs = [0] * np.product(training_env.observation_space.shape)
            rew = 0
            for step in range(episode_length):
                # TODO
                train_batch(
                    model,
                    optimizer,
                    baseline,
                    0,
                    episode,
                    step,
                    batch,
                    tb_logger,
                    opts,
                    rew=rew,
                    edges=acs_edges_dict,
                    agent_nodes=[agent_node]
                )
    else:
        # create model
        ppo_trainer = ppo.PPOTrainer(config=create_trainer_config(outer_configs, trainer_type=ppo, custom_model=True), env=Figure8SquadRLLib)
        print('trainer created')
        # test model
        ppo_trainer.train()
        print('model trained')

'''

        # 
        # Start the actual training loop
        val_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)
        # Generate new training data for each epoch
        training_dataset = baseline.wrap_dataset(problem.make_dataset(
            size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))

'''