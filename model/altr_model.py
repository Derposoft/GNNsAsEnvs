# rl/ai imports
from ray.rllib.agents import ppo
import numpy as np
import json
import os
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

# our code
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
from generate_baseline_metrics import parse_arguments, create_env_config, create_trainer_config
from model.utils import embed_obs_in_map, load_edge_dictionary
from sigma_graph.data.file_manager import set_visibility
from model.utils import get_cost_from_reward

# 3rd party
from attention_routing.nets.attention_model import AttentionModel, set_decode_type
from attention_routing.train import train_batch
from attention_routing.utils.log_utils import log_values
from attention_routing.nets.critic_network import CriticNetwork
from attention_routing.options import get_options
from attention_routing.train import train_epoch, validate, get_inner_model
from attention_routing.reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from attention_routing.nets.attention_model import AttentionModel
from attention_routing.nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from attention_routing.utils import torch_load_cpu, load_problem
from attention_routing.train import clip_grad_norms

from copy import deepcopy

local_opts = None
def initialize_train_artifacts(opts):
    '''
    code mostly from attention_routing/run.py:run(opts)
    repurposed for reinforcement learning here.
    :params None
    :returns optimizer, baseline, lr_scheduler, val_dataset, problem, tb_logger, opts
    '''
    global local_opts
    local_opts = deepcopy(opts)
    # Set the random seed
    torch.manual_seed(opts.seed)
    
    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    config_file = "configs/altr_config.json"
    with open(config_file, 'w') as f:
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
    optimizer = optim.SGD(
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
    
    return model, optimizer, baseline, lr_scheduler

def optimize(optimizer, baseline, reward, ll, TEST_SETTINGS, num_steps=1, attention_input=None):
    local_max_theoretical_reward = TEST_SETTINGS['MAXIMUM_THEORETICAL_REWARD']
    if TEST_SETTINGS['normalize_losses_rewards_by_ep_length']:
        reward /= num_steps
        ll /= num_steps
        local_max_theoretical_reward /= num_steps
    # set costs
    model_cost = get_cost_from_reward(reward)
    bl_val, bl_loss = baseline.eval(attention_input, model_cost) #if bl_val is None else (bl_val, 0) # critic loss
    if TEST_SETTINGS['use_hardcoded_bl'] or attention_input == None:
        bl_val = get_cost_from_reward(local_max_theoretical_reward)
        bl_loss = 0
    #reinforce_loss = -((model_cost - bl_val) * ll).mean()
    reinforce_loss = ((model_cost - bl_val) * ll).mean()
    print(ll)
    print(f'model_cost: {model_cost}, bl_cost: {bl_val}, loss: {reinforce_loss}')

    loss = reinforce_loss + bl_loss
    # perform optimization step
    optimizer.zero_grad()
    loss.backward()
    grad_norms = clip_grad_norms(optimizer.param_groups, local_opts.max_grad_norm)
    optimizer.step()
    return grad_norms, loss
