'''
all outputted metrics can be found and visualized in tensorboard at ~/ray_results.
'''

# general
import argparse
import pickle
import time
import os

# our code
from sigma_graph.envs.figure8.action_lookup import MOVE_LOOKUP, TURN_90_LOOKUP
from sigma_graph.envs.figure8.default_setup import OBS_TOKEN
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
import sigma_graph.envs.figure8.default_setup as default_setup
import model # THIS NEEDS TO BE HERE IN ORDER TO RUN __init__.py!

# algorithms to test
from ray.rllib.agents import dqn
from ray.rllib.agents import pg
from ray.rllib.agents import a3c
from ray.rllib.agents import ppo
from ray.rllib.agents import impala # single-threaded stuff only for now
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.tune.logger import pretty_print

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
                    #"use_mean_embed": config.use_mean_embed,
                    "fixed_start": config.fixed_start,
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

# create trainer configuration
def create_trainer_config(outer_configs, trainer_type=None, custom_model=''):
    # check params
    if custom_model != '':
        if custom_model != 'altr_policy' and custom_model != 'graph_transformer_policy' and custom_model != 'fc_policy':
            raise ValueError('custom_model parameter must be altr_policy or graph_transformer_policy or empty!')
    trainer_types = [dqn, pg, a3c, ppo]
    assert trainer_type != None, f'trainer_type must be one of {trainer_types}'

    # initialize env and required config settings
    setup_env = Figure8SquadRLLib(outer_configs)
    obs_space = setup_env.observation_space
    act_space = setup_env.action_space
    policies = {}
    for agent_id in setup_env.learning_agent:
        policies[str(agent_id)] = (None, obs_space, act_space, {})
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return str(agent_id)
    CUSTOM_DEFAULTS = {
        "custom_model": custom_model,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "map": setup_env.map,
            "nred": outer_configs['n_red'],
            "nblue": outer_configs['n_blue'],
            #"use_mean_embed": outer_configs['use_mean_embed'],
        },
    }
    init_trainer_config = {
        "env": Figure8SquadRLLib,
        "env_config": {
            **outer_configs
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": CUSTOM_DEFAULTS if custom_model != '' else MODEL_DEFAULTS,
        "num_workers": 1,  # parallelism
        "framework": "torch",
        "evaluation_interval": 1,
        "evaluation_num_episodes": 10,
        "evaluation_num_workers": 1,
        "rollout_fragment_length": 50, # 50 for a2c, 200 for everyone else?
        "train_batch_size": 200,
        "log_level": "ERROR"
    }

    # initialize specific trainer type config
    trainer_type_config = {}
    trainer_type_config = trainer_type.DEFAULT_CONFIG.copy()
    trainer_type_config.update(init_trainer_config)
    # TODO tune these with scheduler?
    if custom_model != '':
        trainer_type_config['lr'] = 1e-3
    else:
        trainer_type_config["lr"] = 1e-3

    # merge init config and trainer-specific config and return
    trainer_config = { **init_trainer_config, **trainer_type_config }
    return trainer_config

def train(trainer, model_name, train_time=200, checkpoint_models=True, config=None):
    assert model_name != '', 'you must name your model. please use --name'
    if checkpoint_models: assert config != None, 'configs must not be none if models are being saved.'
    for _ in range(train_time):
        result = trainer.train()
        print(pretty_print(result))
    if checkpoint_models:
        #model_dir = 'checkpoints/'+model_name+str(time.time()) + '/'
        model_dir = 'checkpoints/'+model_name+'/'
        checkpoint_path = trainer.save(checkpoint_dir=model_dir+'model')
        with open(model_dir+'config.pkl', 'wb') as f:
            pickle.dump(config, f)
        with open(model_dir+'checkpoint_path.txt', 'w') as f:
            f.write(checkpoint_path)

# run baseline tests with a few different algorithms
def run_baselines(config, run_default_baseline_metrics=False, train_time=200, checkpoint_models=True, custom_model='graph_transformer_policy'):
    '''
    runs a set of baseline algorithms on the red v blue gym environment using rllib. the
    chosen algorithms are from the following list of algorithms:
    https://docs.ray.io/en/latest/rllib-algorithms.html#available-algorithms-overview

    the only requirements for an algorithm to function with the environment are:
    (a) Continuous Actions - Yes. (because MultiDiscrete counts as continuous :c...
        perhaps we can get rid of this requirement by "flattening" our action space into
        a more simple Discrete action space in the future)
    (b) Multi-Agent - Yes. Because the red v blue is a multi-agent environment.

    experimentally, ppo was the only one that performed/worked well with the gat model. therefore,
    the experiments are all focused around its use.
    '''
    # get env config
    outer_configs, _ = create_env_config(config)
    
    # train
    if run_default_baseline_metrics:
        ppo_config = create_trainer_config(outer_configs, trainer_type=ppo, custom_model='fc_policy')
        ppo_trainer_baseline = ppo.PPOTrainer(config=ppo_config, env=Figure8SquadRLLib)
        train(ppo_trainer_baseline, config.name, train_time, checkpoint_models, ppo_config)
    else:
        ppo_config = create_trainer_config(outer_configs, trainer_type=ppo, custom_model=custom_model)
        ppo_trainer_custom = ppo.PPOTrainer(config=ppo_config, env=Figure8SquadRLLib)
        train(ppo_trainer_custom, config.name, train_time, checkpoint_models, ppo_config)

# parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    # configs for sigma_graph env
    parser.add_argument('--env_path', type=str, default='./libraries/combat_env', help='path of the project root')
    parser.add_argument('--n_red', type=int, default=1, help='numbers of red agent')
    parser.add_argument('--n_blue', type=int, default=1, help='numbers of blue agent')
    parser.add_argument('--n_episode', type=int, default=128, help='numbers of episodes per training cycle')
    parser.add_argument('--max_step', type=int, default=20, help='max step for each episode')
    parser.add_argument('--init_health', type=int, default=20, help='initial HP for all agents')
    # advanced configs
    parser.add_argument('--obs_embed_on', dest="obs_embed", action='store_true', default=OBS_TOKEN['obs_embed'],
                        help='encoded embedding rather than raw one-hot POS')
    parser.add_argument('--obs_dir_off', dest="obs_dir", action='store_false', default=OBS_TOKEN['obs_dir'],
                        help='observation self 4 dir')
    parser.add_argument('--obs_team_off', dest="obs_team", action='store_false', default=OBS_TOKEN['obs_team'],
                        help='observation teammates')
    parser.add_argument('--obs_sight_off', dest="obs_sight", action='store_false', default=OBS_TOKEN['obs_sight'],
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
    parser.add_argument('--penalty_stay', type=int, default=0, help='penalty for take stay action [0: "NOOP"]')
    parser.add_argument('--threshold_blue', default=2)
    parser.add_argument('--threshold_red', default=5)

    # model/training config
    parser.add_argument('--name', default='', help='name this model')
    parser.add_argument('--model', default='graph_transformer', choices=['graph_transformer', 'altr'])
    parser.add_argument('--train_time', type=int, default=200, help='how long to train the model')
    parser.add_argument('--use_mean_embed', type=bool, default=False, help='use mean embeddings vs choose embedding for agent\'s node at inference time')
    parser.add_argument('--run_baselines', type=bool, default=False, help='are we running baselines or actual model?')
    parser.add_argument('--fixed_start', type=int, default=-1, help='where to fix the agent init points when training')


    # testing config
    parser.add_argument('--policy_file', type=str, default='', help='use hardcoded policy from provided policy file')

    return parser

if __name__ == "__main__":
    # parse args and run baselines
    parser = parse_arguments()
    config = parser.parse_args()
    run_baselines(config, run_default_baseline_metrics=config.run_baselines, custom_model=config.model+'_policy', train_time=config.train_time)



'''# junk section; TODO remove

        #ppo_trainer_default = ppo.PPOTrainer(config=create_trainer_config(outer_configs, trainer_type=ppo), env=Figure8SquadRLLib)
        #a2c_trainer_default = a3c.A2CTrainer(config=create_trainer_config(outer_configs, trainer_type=a3c), env=Figure8SquadRLLib)
        #pg_trainer_default = pg.PGTrainer(config=create_trainer_config(outer_configs, trainer_type=pg), env=Figure8SquadRLLib)
        #dqn_trainer_default = dqn.DQNTrainer(config=create_trainer_config(outer_configs, trainer_type=dqn), env=Figure8SquadRLLib)
        #train(ppo_trainer_default, 'ppo_default', train_time, checkpoint_models)
        #train(a2c_trainer_default, 'a2c_default', train_time, checkpoint_models)
        #train(pg_trainer_default)
        #train(dqn_trainer_default)
        #a2c_trainer_custom = a3c.A2CTrainer(config=create_trainer_config(outer_configs, trainer_type=a3c, custom_model=custom_model), env=Figure8SquadRLLib)
        #pg_trainer_custom = pg.PGTrainer(config=create_trainer_config(outer_configs, trainer_type=pg, custom_model=custom_model), env=Figure8SquadRLLib)
        #dqn_trainer_custom = dqn.DQNTrainer(config=create_trainer_config(outer_configs, trainer_type=dqn, custom_model=custom_model), env=Figure8SquadRLLib)
        #a2c_trainer_custom = a3c.A2CTrainer(config=create_trainer_config(outer_configs, trainer_type=a3c, custom_model=custom_model), env=Figure8SquadRLLib)
        #pg_trainer_custom = pg.PGTrainer(config=create_trainer_config(outer_configs, trainer_type=pg, custom_model=custom_model), env=Figure8SquadRLLib)
        #dqn_trainer_custom = dqn.DQNTrainer(config=create_trainer_config(outer_configs, trainer_type=dqn, custom_model=custom_model), env=Figure8SquadRLLib)
        #train(a2c_trainer_custom, 'a2c_custom', train_time, checkpoint_models)
        #train(pg_trainer_custom)
        #train(dqn_trainer_custom)


'''