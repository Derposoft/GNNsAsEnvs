'''
tests models that are outputted by generate_metrics.py.
'''

# general
import pickle
import sys
import time
import torch
import os

# our code
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
import model # THIS NEEDS TO BE HERE IN ORDER TO RUN __init__.py!
from metrics import create_env_config, create_trainer_config, parse_arguments

# algorithms to test
from ray.rllib.agents import ppo

def restore_trainer(checkpoint_path, config):
    '''
    https://docs.ray.io/en/latest/serve/tutorials/rllib.html
    '''
    config['log_level'] = 'ERROR'
    trainer = ppo.PPOTrainer(config, env=Figure8SquadRLLib)
    trainer.restore(checkpoint_path)
    return trainer


# run baseline tests with a few different algorithms
def run_tests(config):
    '''
    runs a set of tests on the models
    '''
    # initialize env
    outer_config, _ = create_env_config(config)
    test_env = Figure8SquadRLLib(outer_config)

    # test all models trained so far
    print('testing...')
    dir = './checkpoints'
    checkpoints = os.listdir(dir)
    for checkpoint in checkpoints:
        model_dir = f'{dir}/{checkpoint}'
        print(f'########## model at: {model_dir} ##########')
        with open(model_dir+'/config.pkl', 'rb') as f:
            trainer_config = pickle.load(f)
        with open(model_dir+'/checkpoint_path.txt', 'r') as f:
            checkpoint_path = f.readlines()[0]
        trainer = restore_trainer(checkpoint_path, trainer_config)
        print('restored')
        # test all possible starting locations for red and print policy for each of location
        for i in range(test_env.map.get_graph_size()):
            obs, _, done = test_env.reset(), 0, False
            for j in range(len(test_env.team_red)):
                test_env.team_red[j].set_location(i+1, test_env.map.get_name_by_index(i+1), 1)
            locations = {}
            for agent in obs.keys():
                locations[agent] = [i+1]
            actions = {}
            # go till either 20 steps or done
            for step in range(20):
                # keep track of actions+locations gone by each agent
                n_action = {}
                for agent in obs.keys():
                    agent_obs = obs[agent]
                    agent_action = trainer.compute_single_action(torch.tensor(agent_obs))
                    if agent not in actions: actions[agent] = []
                    actions[agent].append(agent_action)
                    n_action[agent] = agent_action
                obs, _, done, _ = test_env.step(n_action)
                for agent in obs.keys():
                    locations[agent].append(test_env.team_red[int(agent)].agent_node)
                if done['__all__']:
                    break
            print(f'r1n{i} policy:')
            for agent in obs.keys():
                print(f'agent {agent}A: {actions[agent]}')
                print(f'agent {agent}N: {locations[agent]}')
            
    print('done')

if __name__ == "__main__":
    # parse args and run tests
    parser = parse_arguments()
    config = parser.parse_args()
    run_tests(config)
