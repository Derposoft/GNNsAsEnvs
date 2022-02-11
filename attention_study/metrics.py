# rl/ai imports
from argparse import ArgumentError
import sys
from ray.rllib.agents import ppo
import numpy as np
import json
import os
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

# our code
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
from attention_study.generate_baseline_metrics import parse_arguments, create_env_config, create_trainer_config
from attention_study.model.utils import embed_obs_in_map, load_edge_dictionary, get_probs_mask
from sigma_graph.data.file_manager import set_visibility
from attention_study.model.altr_model import optimize as optimize_altr
from attention_study.model.altr_model import initialize_train_artifacts as initialize_altr
from attention_study.model.graph_transformer_model import optimize as optimize_graph_transformer
from attention_study.model.graph_transformer_model import initialize_train_artifacts as initialize_graph_transformer

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

TEST_SETTINGS = {
    'model_selection': 'altr', # which model are we using? options: 'altr', 'graph_transformer'
    'is_standalone': True, # are we training in rllib, or standalone?
    'is_360_view': True, # can the agent see in all directions at once?
    'is_obs_embedded': False, # are our observations embedded into the graph?
    'is_per_step': False, # do we optimize every step if True, or every episode if False?
    'is_mask_in_model': True, # do we use the mask in the model or after the model?
    'use_hardcoded_bl': True, # subtract off a hardcoded "baseline" value
    'normalize_losses_rewards_by_ep_length': False, # divide losses/rewards/loglosses by ep length?
    'episode_length': 20, # length of each episode
    'num_episodes': 10000, # number of episodes to train for
}
MAXIMUM_THEORETICAL_REWARD = 25

if __name__ == "__main__":
    # init config
    print('creating config')
    parser = parse_arguments()
    config = parser.parse_args()
    outer_configs, n_episodes = create_env_config(config)
    outer_configs['max_step'] = TEST_SETTINGS['episode_length']
    set_visibility(TEST_SETTINGS['is_360_view'])
    tb_logger = TbLogger(os.path.join('logs', "{}_{}".format('redvblue', 27), TEST_SETTINGS['model_selection']))
    
    # train in rllib [NOT CURRENTLY IN USE]
    if not TEST_SETTINGS['is_standalone']:
        # create model
        ppo_trainer = ppo.PPOTrainer(config=create_trainer_config(outer_configs, trainer_type=ppo, custom_model=True), env=Figure8SquadRLLib)
        print('trainer created')
        # test model
        ppo_trainer.train()
        print('model trained')
        sys.exit()
    
    # train model outside of rllib

    # init model and optimizer
    if TEST_SETTINGS['model_selection'] == 'altr':
        opts = get_options()
        model, optimizer, baseline, lr_scheduler = initialize_altr(opts)
        model.train()
        set_decode_type(model, "sampling")
    elif TEST_SETTINGS['model_selection'] == 'graph_transformer':
        model, optimizer, _, lr_scheduler = initialize_graph_transformer()
        model.train()
    else:
        raise ValueError('TEST_SETTINGS model_selection parameter is invalid!')
    
    # init training env
    training_env = Figure8SquadRLLib(outer_configs)
    acs_edges_dict = load_edge_dictionary(training_env.map.g_acs.adj)
    obs = [[0] * np.product(training_env.observation_space.shape)]
    if not TEST_SETTINGS['is_obs_embedded']:
        attention_input = embed_obs_in_map(obs, training_env.map) # can be done here if not embedded
    
    # training loop
    print('training')
    episode_length = training_env.max_step
    num_training_episodes = TEST_SETTINGS['num_episodes']
    total_reward = 0
    total_ll = None
    logged_reward = 0 # for logging
    logged_ll = None
    for episode in range(num_training_episodes):
        training_env.reset();
        agent_node = training_env.team_red[training_env.learning_agent[0]].agent_node # 1-indexed value
        for step in range(episode_length):
            # get model predictions
            if TEST_SETTINGS['is_obs_embedded']:
                attention_input = embed_obs_in_map(obs, training_env.map) # embed obs every time we get a new obs
            
            if TEST_SETTINGS['model_selection'] == 'altr':
                if TEST_SETTINGS['is_mask_in_model']:
                    cost, ll, log_ps = model(attention_input, acs_edges_dict, [agent_node-1], return_log_p=True)
                else:
                    cost, ll, log_ps = model(attention_input, return_log_p=True)
                
                # mask model predictions with our graph edges
                if not TEST_SETTINGS['is_mask_in_model']:
                    mask = get_probs_mask([agent_node-1], attention_input.shape[1], acs_edges_dict)
                    for i in range(len(mask)):
                        log_ps[i][mask[i]] = 0
                
                # move_action decoding. get max prob moves from masked predictions
                features = log_ps # set features for value branch later
                transformed_features = features.clone()
                transformed_features[transformed_features == 0] = -float('inf')
                optimal_destination = torch.argmax(transformed_features, dim=1)
                curr_loc = agent_node
                next_loc = optimal_destination[0].item() + 1
                move_action = 0 if curr_loc == next_loc else training_env.map.g_acs.adj[curr_loc][next_loc]['action']
                look_action = 1 # TODO!!!!!!!! currently uses all-way look
                action = Figure8SquadRLLib.convert_multidiscrete_action_to_discrete(move_action, look_action)
            elif TEST_SETTINGS['model_selection'] == 'graph_transformer':
                pass

            # step through environment to update obs/rew and agent node
            actions = {}
            for a in training_env.learning_agent:
                actions[str(a)] = action
            obs, rew, done, _ = training_env.step(actions)
            agent_node = training_env.team_red[training_env.learning_agent[0]].agent_node

            # collect rewards/losses
            for a in training_env.learning_agent:
                total_reward += rew[str(a)]
            if not total_ll:
                total_ll = ll
            else:
                total_ll += ll
            
            # optimize once per step
            if TEST_SETTINGS['is_per_step']:
                grad_norms, loss = optimize_altr(optimizer, baseline, total_reward, total_ll, TEST_SETTINGS)
                # reset for next iteration
                logged_reward += total_reward
                logged_ll = total_ll if not logged_ll else logged_ll + total_ll
                total_reward = 0
                total_ll = None
            
            # end episode if simulation is done
            if done['__all__']:
                break
        
        # optimize once per episode
        if not TEST_SETTINGS['is_per_step']:
            grad_norms, loss = optimize_altr(optimizer, baseline, total_reward, total_ll, TEST_SETTINGS, episode_length)
            # reset for next iteration
            logged_reward = total_reward
            logged_ll = total_ll
            total_reward = 0
            total_ll = None
        
        # log results
        print('reward', logged_reward)
        # log step in tb for metrics
        #if episode % int(opts.log_step) == 0:
        if episode % 10 == 0:
            logged_reward = torch.tensor(logged_reward, dtype=torch.float32)
            log_values(logged_reward, grad_norms, episode, episode, episode,
                    logged_ll, loss, 0, tb_logger, opts, mode="reward")
        logged_reward = 0
        logged_ll = None
        
