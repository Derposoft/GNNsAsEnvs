'''
tests models that are outputted by generate_metrics.py.
'''

# general
import pickle
import sys
import time
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
    trainer = ppo.PPOTrainer(config, env=Figure8SquadRLLib)
    trainer.restore(checkpoint_path+'/checkpoint_000001/checkpoint-1')
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
        print(f'model at: {model_dir}')
        with open(model_dir+'/config.pkl', 'rb') as f:
            trainer_config = pickle.load(f)
        trainer = restore_trainer(model_dir+'/model', trainer_config)
        print('restored')
        # test all possible starting locations for red and print policy for each of location
        for i in range(test_env.map.get_graph_size()):
            test_env.reset()
            print(f'r1n{i}')
            #test_env.learning_agent[0]
    print('done')

if __name__ == "__main__":
    # parse args and run tests
    parser = parse_arguments()
    config = parser.parse_args()
    run_tests(config)
