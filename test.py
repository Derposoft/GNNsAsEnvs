"""
tests models that are outputted by generate_metrics.py.
"""

# general
from email import policy
import pickle
import sys
import time
import torch
import json
import os

# our code
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
import model # THIS NEEDS TO BE HERE IN ORDER TO RUN __init__.py!
from train import create_env_config, create_trainer_config, parse_arguments, custom_log_creator

# algorithms to test
from ray.rllib.agents import ppo

def restore_trainer(checkpoint_path, config):
    """
    https://docs.ray.io/en/latest/serve/tutorials/rllib.html
    """
    config["log_level"] = "ERROR"
    trainer = ppo.PPOTrainer(config, env=Figure8SquadRLLib, logger_creator=custom_log_creator("junk", custom_dir="./logs"))
    trainer.restore(checkpoint_path)
    return trainer


# run baseline tests with a few different algorithms
def run_tests(config):
    """
    runs a set of tests on the models
    """
    # initialize env
    outer_config, _ = create_env_config(config)
    test_env = Figure8SquadRLLib(outer_config)
    policy_file = {}
    if config.policy_file != "":
        with open(config.policy_file) as f:
            policy_file = json.load(f)
    
    # test all models trained so far
    print("testing...")
    dir = "./checkpoints"
    checkpoints = os.listdir(dir)
    for checkpoint in checkpoints:
        model_dir = f"{dir}/{checkpoint}"
        if not os.path.exists(model_dir+"/config.pkl"): # skip over subdirectories that i might be using for storing old checkpoints
            continue
        print(f"########## model at: {model_dir} ##########")
        with open(model_dir+"/config.pkl", "rb") as f:
            trainer_config = pickle.load(f)
        with open(model_dir+"/checkpoint_path.txt", "r") as f:
            checkpoint_path = f.readlines()[0].lstrip().rstrip()
        trainer = restore_trainer(checkpoint_path, trainer_config)
        print("restored")
        # test all possible starting locations for red and print policy for each of location
        tot_rew_across_all = {}
        for i in range(test_env.map.get_graph_size()):
            obs, done = test_env.reset(), False
            for j in range(len(test_env.team_red)):
                test_env.team_red[j].set_location(i+1, test_env.map.get_name_by_index(i+1), 1)
            actions = {}
            locations = {}
            rew = {}
            hp = {}
            #blue_locations = {} # TODO add location printout for blue as well
            for agent in obs.keys():
                locations[agent] = [i+1]
            # go till either 20 steps or done
            for step in range(20):
                # keep track of actions+locations gone by each agent
                n_action = {}
                for agent in obs.keys():
                    agent_obs = obs[agent]
                    hardcoded_policy_this_step = False
                    if str(agent) in policy_file and str(i+1) in policy_file[str(agent)]:
                        ax_ni_policy = policy_file[str(agent)][str(i+1)]
                        if step < len(ax_ni_policy):
                            agent_action = Figure8SquadRLLib.convert_multidiscrete_action_to_discrete(
                                ax_ni_policy[step][0], ax_ni_policy[step][1]
                            )
                            hardcoded_policy_this_step = True
                    if not hardcoded_policy_this_step:
                        agent_action = trainer.compute_single_action(torch.tensor(agent_obs))
                    if agent not in actions: actions[agent] = []
                    actions[agent].append(agent_action)
                    n_action[agent] = agent_action
                obs, nrew, done, _ = test_env.step(n_action)
                for agent in nrew:
                    if agent not in rew: rew[agent] = 0
                    rew[agent] += nrew[agent]
                for agent in obs.keys():
                    locations[agent].append(test_env.team_red[int(agent)].agent_node)
                for agent in obs.keys():
                    if agent not in hp: hp[agent] = 0
                    hp[agent] = test_env.team_red[int(agent)].health
                    #print("current agent hp:", test_env.team_red[int(agent)].health)
                if done["__all__"]:
                    break
            for agent in obs.keys():
                print(f"agent{agent}n{i+1} policy:")
                print(f"agent {agent}A: {actions[agent]}")
                print(f"agent {agent}N: {locations[agent]}")
                print(f"agent {agent}R: {rew[agent]}")
                print(f"agent {agent}H: {hp[agent]}")
                tot_rew_across_all[agent] = tot_rew_across_all.get(agent, 0) + rew[agent]
                del actions, locations, rew, hp
        print(f"total model reward among all initializations: {tot_rew_across_all}")
            
    print("done")

if __name__ == "__main__":
    # parse args and run tests
    parser = parse_arguments()
    config = parser.parse_args()
    #if "policy" in config:
    #    print(config.policy)
    #    sys.exit()
    run_tests(config)
