import torch.nn as nn
import torch
from sigma_graph.data.graph.skirmish_graph import MapInfo

def embed_obs_in_map(obs: torch.Tensor, map: MapInfo):
    print('starting embedding process')
    print(obs)
    print(f'observation length: {len(obs)}') # in sigma_graph.env.default_setup.get_state_shape()
    print(map.n_name)
    print(map.n_info)
    pass