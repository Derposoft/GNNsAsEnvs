from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.rllib.agents import dqn
import numpy as np
import os

from sigma_graph.envs.figure8.figure8_squad import Figure8Squad
from . import default_setup as env_setup
local_action_move = env_setup.act.MOVE_LOOKUP
local_action_turn = env_setup.act.TURN_90_LOOKUP


# a variant of figure8_squad that can be used by rllib multiagent setups
# reference: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
class Figure8SquadRLLib(Figure8Squad, MultiAgentEnv):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(**config)
        
        # extra values to make graph embedding viable
        num_extra_graph_obs = 0# 5 if self.obs_token["obs_graph"] else 0
        # self.action_space = spaces.MultiDiscrete([len(local_action_move), len(local_action_turn)])
        # "flatten" the above action space into the below discrete action space
        self.action_space = spaces.Discrete(len(local_action_move)*len(local_action_turn))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_shape + num_extra_graph_obs,), dtype=np.int8)
        self.done = set()

    # return an arbitrary encoding from the "flat" action space to the normal action space 0-indexed
    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]
    def convert_multidiscrete_action_to_discrete(move_action, turn_action):
        return turn_action * len(local_action_move) + move_action

    def reset(self):
        _resets = super().reset()
        resets = {}
        for idx in range(len(_resets)):
            resets[str(self.learning_agent[idx])] = _resets[idx]
        self.done = set()
        return resets
    
    def step(self, _n_actions: dict):
        # reference: https://docs.ray.io/en/latest/rllib-env.html#pettingzoo-multi-agent-environments
        # undictify the actions to interface rllib -> env input
        n_actions = []
        for a in self.learning_agent:
            if str(a) in _n_actions:
                n_actions.append(self.convert_discrete_action_to_multidiscrete(_n_actions.get(str(a))))
            else:
                n_actions.append(self.convert_discrete_action_to_multidiscrete(0))
        _obs, _rew, _done, _ = super().step(n_actions)

        # dictify the observations to interface env output -> rllib
        obs, rew, done = {}, {}, {}
        all_done = True
        for a_id in self.learning_agent:
            if a_id in self.done:
                continue
            # obs for graph nns
            '''if self.obs_token["obs_graph"]:
                # info to allow for easy obs dissection for graph embedding
                self_shape, red_shape, blue_shape = env_setup.get_state_shapes(self.map.get_graph_size(), self.num_red, self.num_blue, env_setup.OBS_TOKEN)
                # get info about self
                n_red = self.num_red
                n_blue = self.num_blue
                # return obs that is easy for node embedding
                #_obs_a = _obs[a_id] #[self_shape, red_shape, blue_shape, n_red, n_blue] + _obs[a_id]
                _obs_a = np.hstack([[self_shape, red_shape, blue_shape, n_red, n_blue], _obs[a_id]])
                obs[str(a_id)] = _obs_a
            # obs for normal ff nns
            else:'''
            obs[str(a_id)] = _obs[a_id]
            rew[str(a_id)] = _rew[a_id]
            done[str(a_id)] = _done[a_id]
            if _done[a_id]:
                self.done.add(a_id)
            # for some reason in rllib MARL __all__ must be included in 'done' dict
            all_done = all_done and _done[a_id]
        done['__all__'] = all_done
        
        return obs, rew, done, {}


# create env configuration
def create_env_config(config):
    n_episodes = config.n_episode
    # init_red and init_blue should have number of agents dictionary elements if you want to specify it
    # [!!] remember to update this dict if adding new args in parser
    outer_configs = {
        "env_path": config.env_path, "max_step": config.max_step, "act_masked": config.act_masked,
        "n_red": config.n_red, "n_blue": config.n_blue,
        "init_red": config.init_red, "init_blue": config.init_blue,
        "init_health_red": config.init_health, "init_health_blue": config.init_health,
        "obs_embed": config.obs_embed, "obs_dir": config.obs_dir, "obs_team": config.obs_team,
        "obs_sight": config.obs_sight,
        "log_on": config.log_on, "log_path": config.log_path,
        # "reward_step_on": False, "reward_episode_on": True, "episode_decay_soft": True,
        # "health_lookup": {"type": "table", "reward": [8, 4, 2, 0], "damage": [0, 1, 2, 100]},
        # "faster_lookup": {"type": "none"},
        "fixed_start": config.fixed_start,
        #"aggregation_fn": config.aggregation_fn,
        #"hidden_size": config.hidden_size,
        #"is_hybrid": config.is_hybrid,
        #"conv_type": config.conv_type,
    }
    ## i.e. init_red "pos": tuple(x, z) or "L"/"R" region of the map
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
    # make dqn trainer
    outer_configs, n_episodes = create_env_config(config)
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

    print('dqn trainer loaded.')
    # train
    print('training...')
    result = dqn_trainer.train()

    