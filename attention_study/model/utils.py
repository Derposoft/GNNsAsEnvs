from os import posix_fadvise
import sys
import torch.nn as nn
import torch
from copy import deepcopy
import sigma_graph.envs.figure8.default_setup as env_setup
from sigma_graph.data.graph.skirmish_graph import MapInfo

# constants/helper functions
NETWORK_SETTINGS = {
    'has_final_layer': False,
    'use_altr_model': True,
    'use_s2v': False,
}
SUPPRESS_WARNINGS = {
    'embed': False,
    'decode': False,
}
GRAPH_OBS_TOKEN = {
    'obs_embed': False,
    'embedding_size': 4,
}
NODE_EMBEDDING_SIZE = 4
EMBED_IDX = {
    'is_agent_pos': 2,
    'agent_dir': 3, # 0 if agent is not here
    'is_red_here': 4,
    'is_blue_here': 5,
}
def ERROR_MSG(e): return f'ERROR READING OBS: {e}'

# TODO read obs using obs_token instead of hardcoding.
#      figure8_squad.py:_update():line ~250
def embed_obs_in_map(obs: torch.Tensor, map: MapInfo):
    """
    obs: a batch of inputs
    map: a MapInfo object

    graph embedding:
    [[x, y, agent_is_here, agent_dir, is_red_here, is_blue_here],
    [...],
    ...]
    """
    print('starting embedding process')
    # init node embeddings
    pos_obs_size = map.get_graph_size()
    batch_size = len(obs)
    g = []
    for i in range(pos_obs_size):
        g_ij = list(map.n_info[i + 1])
        if GRAPH_OBS_TOKEN['obs_embed']:
            g_ij += [0] * GRAPH_OBS_TOKEN['embedding_size']
        g.append(g_ij)
    # embed nodes using obs
    node_embeddings = []
    for i in range(batch_size):
        g_i = deepcopy(g)
        if GRAPH_OBS_TOKEN['obs_embed']:
            obs_i = obs[i]
            embed(obs_i, g_i)
        node_embeddings.append(g_i)
    node_embeddings = torch.tensor(node_embeddings)#.cuda()

    # TODO edges?
    edges = None
    return node_embeddings

def embed(obs, g):
    """
    obs: a single input
    g: nodes of a graph with empty embeddings

    creates graph embedding !!!!!IN PLACE!!!!!:
    [[x, y, agent_is_here, agent_dir, is_red_here, is_blue_here],
    [...],
    ...]

    returns None
    """
    global SUPPRESS_WARNINGS
    # get obs parts
    pos_obs_size = len(g)
    look_dir_shape = len(env_setup.ACT_LOOK_DIR)
    self_shape, red_shape, blue_shape, n_red, n_blue = obs[:5].int().tolist()
    if self_shape < pos_obs_size or red_shape < pos_obs_size or blue_shape < pos_obs_size:
        if SUPPRESS_WARNINGS['embed']:
            print(ERROR_MSG('test batch detected while embedding. skipping embed and suppressing this warning.'))
            SUPPRESS_WARNINGS['embed'] = True
        return
    #assert(red_shape % n_red == 0)
    #assert(blue_shape % n_blue == 0)
    obs = obs[5:]
    self_obs = obs[:self_shape]
    blue_obs = obs[self_shape:(self_shape+blue_shape)]
    red_obs = obs[(self_shape+blue_shape):(self_shape+blue_shape+red_shape)]
    
    # embed self info
    # embed location
    _node = get_loc(self_obs, pos_obs_size)
    if _node == -1:
        print(ERROR_MSG('agent not found ('))
    g[_node][EMBED_IDX['is_agent_pos']] = 1
    # embed direction TODO embed direction in one hot instead of int
    _dir = self_obs[pos_obs_size:(pos_obs_size+look_dir_shape)]
    g[_node][EMBED_IDX['agent_dir']] = int(''.join(_dir), base=2)

    # embed blue info
    # embed locations
    for i in range(pos_obs_size):
        if blue_obs[i]:
            g[i][EMBED_IDX['is_blue_here']] = 1

    # embed red info
    # embed locations
    for i in range(pos_obs_size):
        if red_obs[i]:
            g[i][EMBED_IDX['is_red_here']] = 1

# get location of an agent given one-hot positional encoding on graph (1-indexed)
def get_loc(one_hot_graph, graph_size, default=0):
    global SUPPRESS_WARNINGS
    for i in range(graph_size):
        if one_hot_graph[i]:
            return i
    if not SUPPRESS_WARNINGS['decode']:
        print(f'test batch detected while decoding. agent not found. returning default={default} and suppressing this warning.')
        SUPPRESS_WARNINGS['decode'] = True
    return default

# load edge dictionary from a map (0-indexed)
def load_edge_dictionary(map_edges):
    '''
    :param map_edges: edges from a graph from MapInfo. input should be 1-indexed MapInfo map_edge dictionary.
    :return the 0-indexed edge dictionary for quick lookups.
    '''
    # create initial edge_array and TODO edge_to_action mappings
    edge_array = []
    for k, v in zip(map_edges.keys(), map_edges.values()):
        edge_array += [[k-1, vi-1] for vi in v.keys()]
    
    # create edge_dictionary
    edge_dictionary = {}
    for edge in edge_array:
        if edge[0] not in edge_dictionary: edge_dictionary[edge[0]] = set([])
        edge_dictionary[edge[0]].add(edge[1])
    
    # TODO create edge_to_action dictionary
    return edge_dictionary