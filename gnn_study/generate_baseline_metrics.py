'''
a variant of simple_figure8squad.py that rllib can interface with. all outputted metrics
can be found and visualized in tensorboard at ~/ray_results.
'''

# general import
from numpy.core.numeric import outer
from ray.rllib.models.catalog import MODEL_DEFAULTS
from sigma_graph.envs.figure8.action_lookup import MOVE_LOOKUP, TURN_90_LOOKUP
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
import argparse
import time
import gym
import os

# visualize training
import tensorboard as tb
from ray.tune.logger import pretty_print

# algorithms to test
from ray.rllib.agents import dqn
from ray.rllib.agents import pg
from ray.rllib.agents import a3c
from ray.rllib.agents import ppo
from ray.rllib.agents import impala # single-threaded stuff only for now

# tuning
from ray.tune.stopper import TimeoutStopper
from ray import tune
from ray.rllib.models import ModelCatalog

# print current agent states
def print_agents(env):
    # print("Step #{}/{}".format(env.step_counter, env.max_step))
    print("## Team red has {} agent(s)".format(env.num_red))
    for _i in range(env.num_red):
        _id = env.team_red[_i].get_id()
        _node, _dir = env.team_red[_i].get_pos_dir()
        _look = MOVE_LOOKUP[_dir]
        _hp = env.team_red[_i].get_health()
        _dp = env.team_red[_i].damage_total()
        print("# Agent red #{} @node: <{}> dir: <{}:\'{}\'> "
              "health: <{}> damage: <{}>".format(_id, _node, _dir, _look, _hp, _dp))
        print("mask: {}\nobs: {}".format(env.action_mask[_i], env.states[_i]))
    print("## Team blue has {} agent(s)".format(env.num_blue))
    for _i in range(env.num_blue):
        _id = env.team_blue[_i].get_id()
        _idx = env.team_blue[_i].get_index()
        _node, _dir = env.team_blue[_i].get_pos_dir()
        _look = MOVE_LOOKUP[_dir]
        _hp = env.team_blue[_i].get_health()
        _end = env.team_blue[_i].get_end_step()
        print("# Agent blue #{} at pos_index #{} @node: <{}> dir: <{}:\'{}\'> "
              "health: <{}> death_step: <{}>\n".format(_id, _idx, _node, _dir, _look, _hp, _end))

# create env configuration
def create_env_config(config):
    n_episodes = config.n_episode
    # init_red and init_blue should have number of agents dictionary elements if you want to specify it
    # [!!] remember to update this dict if adding new args in parser
    outer_configs = {"env_path": config.env_path, "max_step": config.max_step, "act_masked": config.act_masked,
                    "n_red": config.n_red, "n_blue": config.n_blue,
                    "init_red": config.init_red, "init_blue": config.init_blue,
                    "init_health_red": config.init_health, "init_health_blue": config.init_health,
                    "obs_embed": config.obs_embed, "obs_dir": config.obs_dir, "obs_team": config.obs_team,
                    "obs_sight": config.obs_sight,
                    "log_on": config.log_on, "log_path": config.log_path,
                    # "reward_step_on": False, "reward_episode_on": True, "episode_decay_soft": True,
                    # "health_lookup": {"type": "table", "reward": [8, 4, 2, 0], "damage": [0, 1, 2, 100]},
                    # "faster_lookup": {"type": "none"},
                    }
    ## i.e. init_red 'pos': tuple(x, z) or "L"/"R" region of the map
    # "init_red": [{"pos": (11, 1), "dir": 1}, {"pos": None}, {"pos": "L", "dir": None}]
    if hasattr(config, "penalty_stay"):
        outer_configs["penalty_stay"] = config.penalty_stay
    if hasattr(config, "threshold_blue"):
        outer_configs["threshold_damage_2_blue"] = config.threshold_blue
    if hasattr(config, "threshold_red"):
        outer_configs["threshold_damage_2_red"] = config.threshold_red
    return outer_configs, n_episodes
# run baseline tests with a few different algorithms
def run_baselines(config):
    '''
    runs a set of baseline algorithms on the red v blue gym environment using rllib. the
    chosen algorithms are from the following list of algorithms:
    https://docs.ray.io/en/latest/rllib-algorithms.html#available-algorithms-overview

    the only requirements for an algorithm to function with the environment are:
    (a) Continuous Actions - Yes. (because MultiDiscrete counts as continuous :c...
        perhaps we can get rid of this requirement by "flattening" our action space into
        a more simple Discrete action space in the future)
    (b) Multi-Agent - Yes. Because the red v blue is a multi-agent environment.

    the "STEPs" done in individual functions are mostly dont that way only so they can
    be minimized in an IDE/text editor (like vs code) for cleanliness, for no other reason
    than the fact that i like it that way.

    '''
    # STEP 1: env config construction
    outer_configs, n_episodes = create_env_config(config)
    
    # STEP 2: make rllib configs and trainers

    # for pg
    def create_dqn_config(outer_configs):
        dqn_extra_config_settings = {
            "env": Figure8SquadRLLib,
            "env_config": {
                **outer_configs
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": MODEL_DEFAULTS,
            "num_workers": 1,  # parallelism
            "framework": "torch",
            "evaluation_interval": 1,
            "evaluation_num_episodes": 10,
            "evaluation_num_workers": 1,
            "rollout_fragment_length": 200,
            "train_batch_size": 200
        }
        dqn_config = dqn.DEFAULT_CONFIG.copy()
        dqn_config.update(dqn_extra_config_settings)
        dqn_config["lr"] = 1e-3
        return dqn_config
    dqn_trainer = dqn.DQNTrainer(config=create_dqn_config(outer_configs), env=Figure8SquadRLLib)
    print('dqn trainer loaded...')
    
    # for pg
    def create_pg_config(outer_configs):
        pg_extra_config_settings = {
            "env": Figure8SquadRLLib,
            "env_config": {
                **outer_configs
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": MODEL_DEFAULTS,
            "num_workers": 1,  # parallelism
            "framework": "torch",
            "evaluation_interval": 1,
            "evaluation_num_episodes": 10,
            "evaluation_num_workers": 1,
            "rollout_fragment_length": 200,
            "train_batch_size": 200
        }
        pg_config = pg.DEFAULT_CONFIG.copy()
        pg_config.update(pg_extra_config_settings)
        pg_config["lr"] = 1e-3
        return pg_config
    pg_trainer = pg.PGTrainer(config=create_pg_config(outer_configs), env=Figure8SquadRLLib)
    print('pg trainer loaded...')

    # for impala
    def create_a2c_config(outer_configs):
        a2c_extra_config_settings = {
            "env": Figure8SquadRLLib,
            "env_config": {
                **outer_configs
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": MODEL_DEFAULTS,
            "num_workers": 1,  # parallelism
            "framework": "torch",
            "evaluation_interval": 1,
            "evaluation_num_episodes": 10,
            "evaluation_num_workers": 1,
            "rollout_fragment_length": 50,
            "train_batch_size": 200
        }
        a2c_config = a3c.DEFAULT_CONFIG.copy()
        a2c_config.update(a2c_extra_config_settings)
        a2c_config["lr"] = 1e-3
        return a2c_config
    a2c_trainer = a3c.A2CTrainer(config=create_a2c_config(outer_configs), env=Figure8SquadRLLib)
    print('a2c trainer loaded...')

    # for ppo
    def create_ppo_config(outer_configs):
        # policy mapping function
        # from https://medium.com/@vermashresth/craft-and-solve-multi-agent-problems-using-rllib-and-tensorforce-a3bd1bb6f556
        setup_env = gym.make('figure8squad-v3', **outer_configs)
        obs_space = setup_env.observation_space
        act_space = setup_env.action_space
        policies = {}
        for agent_id in setup_env.learning_agent:
            policies[str(agent_id)] = (None, obs_space, act_space, {})
        #policies['default_policy'] = (None, obs_space, act_space, {}) # necessary for impala
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return str(agent_id)
        # create trainer config
        ppo_extra_config_settings = {
            "env": Figure8SquadRLLib,
            "env_config": {
                **outer_configs
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": MODEL_DEFAULTS,
            "num_workers": 1,  # parallelism
            "framework": "torch",
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "evaluation_interval": 1,
            "evaluation_num_episodes": 10,
            "evaluation_num_workers": 1,
            "rollout_fragment_length": 200,
            "train_batch_size": 200
        }
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(ppo_extra_config_settings)
        ppo_config["lr"] = 1e-3 # fixed lr instead of schedule, tune this
        return ppo_config
    ppo_trainer = ppo.PPOTrainer(config=create_ppo_config(outer_configs), env=Figure8SquadRLLib)
    print('ppo trainer loaded...')

    # STEP 3: train each trainer
    max_train_seconds = 60*15 # train each trainer for exactly 15 min
    print('beginning training.')
    def train(trainer):
        start = time.time()
        while(True):
            result = trainer.train()
            print(pretty_print(result))
            if (time.time() - start) > max_train_seconds: break
        trainer.save(checkpoint_dir='model_checkpoints/'+str(type(trainer)))
    # train dqn
    print('training dqn')
    train(dqn_trainer)
    # train pg
    print('training pg')
    train(pg_trainer)
    # train impala
    print('training impala')
    train(a2c_trainer)
    # train ppo
    print('training ppo')
    train(ppo_trainer)

# parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    # basic configs
    parser.add_argument('--env_path', type=str, default='../', help='path of the project root')
    parser.add_argument('--n_red', type=int, default=2, help='numbers of red agent')
    parser.add_argument('--n_blue', type=int, default=1, help='numbers of blue agent')
    parser.add_argument('--n_episode', type=int, default=128, help='numbers of episodes per training cycle')
    parser.add_argument('--max_step', type=int, default=20, help='max step for each episode')
    parser.add_argument('--init_health', type=int, default=20, help='initial HP for all agents')
    # advanced configs
    parser.add_argument('--obs_embed_on', dest="obs_embed", action='store_true', default=False,
                        help='encoded embedding rather than raw one-hot POS')
    parser.add_argument('--obs_dir_off', dest="obs_dir", action='store_false', default=True,
                        help='observation self 4 dir')
    parser.add_argument('--obs_team_off', dest="obs_team", action='store_false', default=True,
                        help='observation teammates')
    parser.add_argument('--obs_sight_off', dest="obs_sight", action='store_false', default=True,
                        help='observation in sight indicators')
    parser.add_argument('--act_masked_off', dest="act_masked", action='store_false', default=True,
                        help='invalid action masking')
    parser.add_argument('--init_red', type=list, default=None, help='set init "pos" and "dir" for team red')
    parser.add_argument('--init_blue', type=list, default=None, help='set init "route" and "idx" for team blue')
    parser.add_argument('--log_on', dest="log_on", action='store_true', default=False, help='generate verbose logs')
    parser.add_argument('--log_path', type=str, default='logs/temp/', help='relative path to the project root')
    ''' feel free to add more parser args [!!] keep in mind to update the 'outer_configs' if new args been added here
        All other valid config arguments including {
            _graph_args = {"map_id": 'S', "load_pickle": True}
            _config_args = ["damage_maximum", "damage_threshold_red", "damage_threshold_blue"]
            INTERACT_LOOKUP = {
                "sight_range": -1,  # -1 for unlimited range
                "engage_range": 25,
                "engage_behavior": {"damage": 1, "probability": 1.0},
            }
            INIT_LOGS = {
                "log_on": False, "log_path": "logs/", "log_prefix": "log_", "log_overview": "reward_episodes.txt",
                "log_verbose": False, "log_plot": False, "log_save": True,
            }
        }
    '''
    # additional (comment out if not needed)
    parser.add_argument('--penalty_stay', type=int, default=0, help='penalty for take stay action [0: "NOOP"]')
    parser.add_argument('--threshold_blue', default=2)
    parser.add_argument('--threshold_red', default=5)
    return parser

if __name__ == "__main__":
    # STEP 0: parse cmdline args
    parser = parse_arguments()
    # run baselines
    config = parser.parse_args()
    run_baselines(config)
