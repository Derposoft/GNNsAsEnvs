from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.rllib.agents import dqn
import numpy as np
import os

import sys
from .env_scout_mission_std import ScoutMissionStd

#from . import default_setup as env_setup
#local_action_move = env_setup.act.MOVE_LOOKUP
#local_action_turn = env_setup.act.TURN_90_LOOKUP
from . import action_lookup as env_setup
local_action_move = env_setup.MOVE_LOOKUP
local_action_turn = env_setup.TURN_3_LOOKUP


# a variant of figure8_squad that can be used by rllib multiagent setups
# reference: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
class ScoutMissionStdRLLib(ScoutMissionStd, MultiAgentEnv):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(**config)
        
        # extra values to make graph embedding viable
        self.action_space = self.action_space[0] #spaces.Tuple(self.action_space)
        self.observation_space = self.observation_space[0] # spaces.Tuple(self.observation_space)
        self.done = set()

    # return an arbitrary encoding from the "flat" action space to the normal action space 0-indexed
    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]
    def convert_multidiscrete_action_to_discrete(move_action, turn_action):
        return turn_action * len(local_action_move) + move_action

    def reset(self):
        """
        :returns: dictionary of agent_id -> reset observation
        """
        super().reset()
        self.done = set()
        return self.states.dump_dict()[0]
    
    def step(self, _n_actions: dict):
        n_actions = []
        for _id in self.states.name_list:
            if _id in _n_actions:
                n_actions.append(_n_actions[_id])
            else:
                n_actions.append(self.action_space.sample())
        super().step(n_actions)
        obs, rew, done = self.states.dump_dict()
        all_done = True
        for k in done:
            if done[k]:
                self.done.add(k)
            all_done = all_done and done[k]
        done['__all__'] = all_done

        # make sure to only report done ids once
        for id in self.done:
            done.pop(id)
        return obs, rew, done, {}
