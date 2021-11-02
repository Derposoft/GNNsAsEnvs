from sigma_graph.envs.figure8.figure8_squad import Figure8Squad
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym import spaces
import numpy as np

from . import default_setup as env_setup
local_action_move = env_setup.act.MOVE_LOOKUP
local_action_turn = env_setup.act.TURN_90_LOOKUP

# a variant of figure8_squad that can be used by rllib multiagent setups
# reference: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
class Figure8SquadRLLib(Figure8Squad, MultiAgentEnv):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(**config)

        # self.action_space = spaces.MultiDiscrete([len(local_action_move), len(local_action_turn)])
        # "flatten" the above action space into the below discrete action space
        self.action_space = spaces.Discrete(len(local_action_move)*len(local_action_turn))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_shape,), dtype=np.int8)

    # return an arbitrary encoding from the "flat" action space to the normal action space
    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]

    def reset(self):
        _resets = super().reset()
        resets = {}
        for idx in range(len(_resets)):
            resets[str(self.learning_agent[idx])] = _resets[idx]
        return resets

    def step(self, _n_actions: dict):
        # undictify the actions to interface with rllib
        n_actions = []
        for a in self.learning_agent:
            n_actions.append(self.convert_discrete_action_to_multidiscrete(_n_actions.get(str(a))))
        _obs, _rew, _done, _ = super().step(n_actions)

        # dictify the observations to interface with rllib
        # reference: https://docs.ray.io/en/latest/rllib-env.html#pettingzoo-multi-agent-environments
        obs, rew, done = {}, {}, {}
        all_done = True
        for a_id in self.learning_agent:
            obs[str(a_id)] = _obs[a_id]
            rew[str(a_id)] = _rew[a_id]
            done[str(a_id)] = _done[a_id]
            # for some reason in rllib MARL __all__ must be included in 'done' dict
            all_done = all_done and _done[a_id]
        done['__all__'] = all_done
        
        return obs, rew, done, {}
