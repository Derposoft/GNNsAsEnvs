import sys
import torch.nn as nn
import torch
from copy import deepcopy
from sigma_graph.data.graph.skirmish_graph import MapInfo

# TODO read obs using obs_token instead of hardcoding.
#      figure8_squad.py:_update():line ~250
NODE_EMBEDDING_SIZE = 4
def ERROR_MSG(e): return f'ERROR READING OBS: {e}'
def embed_obs_in_map(obs: torch.Tensor, map: MapInfo):
    """
    obs: a batch of inputs
    map: a MapInfo object

    graph embedding:
    [[x, y, agent_is_here, agent_dir, num_red_here, num_blue_here],
    [...],
    ...]
    """
    print('starting embedding process')
    # init node embeddings
    pos_obs_size = map.get_graph_size()
    batch_size = len(obs)
    g = []
    for i in range(pos_obs_size):
        g.append(list(map.n_info[i + 1] + [0] * NODE_EMBEDDING_SIZE))
    # embed nodes using obs
    node_embeddings = []
    for i in range(batch_size):
        g_i = deepcopy(g)
        obs_i = obs[i]
        embed(obs_i, g_i)
        node_embeddings.append(g_i)
    node_embeddings = torch.tensor(node_embeddings).cuda()
    # TODO get edges
    print(node_embeddings)
    return node_embeddings

def embed(obs, g):
    # get obs parts
    self_shape, red_shape, blue_shape, n_red, n_blue = obs[:5]
    assert(red_shape % n_red == 0)
    assert(blue_shape % n_blue == 0)
    obs = obs[5:]
    self_obs = obs[:self_shape]
    blue_obs = obs[self_shape:(self_shape+blue_shape)]
    red_obs = obs[(self_shape+blue_shape):(self_shape+blue_shape+red_shape)]
    
    # GET INFO OF EACH AGENT FROM SELF PART AND RED PART AND BLU PART
    
    # 1. get self positions of agents from observations in batch
    self_pos = get_self_pos(obs, map)

    # 2. get blue team info from obs
    blue_pos = get_blue_pos(obs, map)

    # 3. get red team info from obs
    for i in range(len(red_obs)):
        if red_obs[i]:
            g[i][]
    red_pos = get_red_pos(obs, map)

    pass

def get_self_pos(obs, map):
    """
    obs: a batch of inputs
    map: a MapInfo object
    """
    pos_obs_size = map.get_graph_size()
    positions = []
    for i in range(len(obs)):
        for j in range(pos_obs_size):
            # red agent is here
            if obs[i][j]:
                positions.append(j)
                break
        if positions[-1] == -1:
            print(ERROR_MSG('agent not found'))
    return positions
