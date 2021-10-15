from sigma_graph.envs.figure8.figure8_squad import Figure8Squad

# a variant of figure8_squad that can be used by rllib multiagent setups
# copied mostly from https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py
class Figure8SquadRLLib(Figure8Squad):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(**config)

    def reset(self):
        _resets = super().reset()
        resets = {}
        for idx in range(len(_resets)):
            resets[self.learning_agent[idx]] = _resets[idx]
        return resets

    def step(self, n_actions):
        _obs, _rew, _done, _ = super().step(n_actions)
        # transform figure8 base class outputs to dictionaries to interface with rllib
        # reference: https://docs.ray.io/en/latest/rllib-env.html#pettingzoo-multi-agent-environments
        obs, rew, done = {}, {}, {}
        num_agents = len(_obs)
        for idx in range(num_agents):
            obs[self.learning_agent[idx]] = _obs[idx]
            rew[self.learning_agent[idx]] = _rew[idx]
            done[self.learning_agent[idx]] = _done[idx]
        
        return obs, rew, done, {}


'''
from . import default_setup as env_setup
from . import figure8_squad as base_env

class Figure8SquadRLLib(base_env.Figure8Squad, MultiAgentEnv):
    def __init__(self, env_config):
        import json
        print("IN RLLIB CLASS", json.dumps(env_config))
        #_max_step=40, _n_red=2, _n_blue=1, kwargs = env_config

        base_env.Figure8Squad.__init__(self, **env_config)
        MultiAgentEnv.__init__(self)

    @override(MultiAgentEnv)
    def step(self, n_actions):
        # transform figure8 base class outputs to dictionaries to interface with rllib
        # reference: https://docs.ray.io/en/latest/rllib-env.html#pettingzoo-multi-agent-environments
        _obs, _rew, _done, _ = super().step(n_actions)
        obs, rew, done = {}, {}, {}
        num_agents = len(_obs)
        for idx in range(num_agents):
            obs[str(idx)] = _obs[idx]
            rew[str(idx)] = _rew[idx]
            done[str(idx)] = _done[idx]
        # return {} for final parameter as that is what figure8 base class does
        return obs, rew, done, {}
'''
