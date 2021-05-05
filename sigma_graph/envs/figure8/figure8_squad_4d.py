import numpy as np
import sys
import os

from .maps.skirmish_graph import MapInfo, RouteInfo

import gym
from gym import spaces


class Figure8Squad4Dir(gym.Env):
    def __init__(self, env_map='S', n_red=2, health=5, damage=1, max_step=40):
        # load connectivity & visibility graphs and blue patrol route
        self.map_info = MapInfo()
        self.pat_info = RouteInfo()

    def reset(self):
        self.step_counter = 0

    def step(self, n_actions):
        pass