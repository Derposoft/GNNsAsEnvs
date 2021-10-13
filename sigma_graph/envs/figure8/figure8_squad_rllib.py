from . import default_setup as env_setup
from . import figure8_squad as base_env

from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.annotations import override

# a variant of figure8_squad_4d that can be used by rllib multiagent setups
class Figure8Squad(base_env.Figure8Squad, MultiAgentEnv):
    def __init__(self):
        pass

    @override(MultiAgentEnv)
    def step(self, n_actions):
        # transform figure8 base class outputs to dictionaries to interface with rllib
        # reference: https://docs.ray.io/en/latest/rllib-env.html#pettingzoo-multi-agent-environments
        _obs, _rew, _done, _ = super(base_env.Figure8Squad).step(n_actions)
        obs, rew, done = {}, {}, {}
        num_agents = len(_obs)
        for idx in range(num_agents):
            obs[str(idx)] = _obs[idx]
            rew[str(idx)] = _rew[idx]
            done[str(idx)] = _done[idx]
        # return {} for final parameter as that is what figure8 base class does
        return obs, rew, done, {}
    

