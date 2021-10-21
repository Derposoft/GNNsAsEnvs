from sigma_graph.envs.figure8.figure8_squad import Figure8Squad
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# a variant of figure8_squad that can be used by rllib multiagent setups
# reference: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
class Figure8SquadRLLib(Figure8Squad, MultiAgentEnv):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(**config)

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
            n_actions.append(_n_actions.get(str(a)))
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
