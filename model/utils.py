from collections import deque
import time
import numpy as np
import torch
import sigma_graph.envs.figure8.default_setup as env_setup
from sigma_graph.data.graph.skirmish_graph import MapInfo
from torchinfo import summary
from ray.rllib.models.torch.misc import SlimFC, normc_initializer 


# constants/helper functions
NETWORK_SETTINGS = {
    "has_final_layer": True,
    "use_altr_model": False,
    #"use_s2v": False,
}
SUPPRESS_WARNINGS = {
    "embed": False,
    "embed_noshapes": False,
    "decode": False,
}
GRAPH_OBS_TOKEN = {
    "embedding_size": 5,#7, #10
    "obs_embed": True,
    "embed_pos": False,
    "embed_opt": True,
}
NODE_EMBED_SIZE = (
    GRAPH_OBS_TOKEN["embedding_size"]
    + (2 if GRAPH_OBS_TOKEN["embed_pos"] else 0)
    + (1 if GRAPH_OBS_TOKEN["embed_opt"] else 0)
)
OPT_SETTINGS = {
    "flanking": True, # does positioning on this node consistute "flanking" the enemy?
}
def ERROR_MSG(e): return f"ERROR READING OBS: {e}"
VIEW_DEGS = {
    "view_1deg_away": None,
    "view_2deg_away": None,
    "view_3deg_away": None,
}
MOVE_DEGS = {
    "move_1deg_away": None,
    "move_2deg_away": None,
    "move_3deg_away": None,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO read obs using obs_token instead of hardcoding.
#      figure8_squad.py:_update():line ~250
def efficient_embed_obs_in_map(obs: torch.Tensor, map: MapInfo, obs_shapes=None):
    """
    :param obs: observation from combat_env gym
    :param map: MapInfo object from inside of combat_env gym (for graph connectivity info)
    :param obs_shapes: info used to partition obs into self/blue team/red team observations
    :return node embeddings for each node of the move graph in map, using obs.
        GRAPH_EMBEDDING=TRUE must be true in the default_env setup for the combat_env.
        the new graph embedding looks as follows:
        [[
            agent_x  (if GRAPH_OBS_TOKEN["embed_pos"] = True),
            agent_y  (if GRAPH_OBS_TOKEN["embed_pos"] = True),
            agent_is_here,
            num_red_here,
            num_blue_here,
            can_red_go_here_t,
            can_blue_see_here_t,
            external_optimization (if GRAPH_OBS_TOKEN["embed_opt"] = True)
        ],
        [...],
        ...]
    """
    # initialize node embeddings tensor
    global SUPPRESS_WARNINGS
    pos_obs_size = map.get_graph_size()
    batch_size = len(obs)
    node_embeddings = torch.zeros(batch_size, pos_obs_size, NODE_EMBED_SIZE) # TODO changed +1 node for a dummy node that we'll use when needed

    # embed x,y
    if GRAPH_OBS_TOKEN["embed_pos"]:
        pos_emb = torch.zeros(pos_obs_size+1, 2) # x,y coordinates
        for i in range(pos_obs_size):
            pos_emb[i,:] = torch.FloatTensor(map.n_info[i + 1])
        # normalize x,y
        min_x, min_y = torch.min(pos_emb[:,0]), torch.min(pos_emb[:,1])
        pos_emb[:,0] -= min_x
        pos_emb[:,1] -= min_y
        node_embeddings /= torch.max(pos_emb)
        node_embeddings[:,:,-3:-1] += pos_emb

    # embed rest of features
    for i in range(batch_size):
        # these default behaviors are required because rllib provides zerod test inputs
        if not obs_shapes:
            print("shapes not provided. returning")
            if not SUPPRESS_WARNINGS["embed"]:
                print(ERROR_MSG("shapes not provided. skipping embed and suppressing this warning."))
                SUPPRESS_WARNINGS["embed_noshapes"] = True
        self_shape, red_shape, blue_shape, n_red, n_blue = obs_shapes
        if self_shape < pos_obs_size or red_shape < pos_obs_size or blue_shape < pos_obs_size:
            if not SUPPRESS_WARNINGS["embed"]:
                print(ERROR_MSG("test batch detected while embedding. skipping embed and suppressing this warning."))
                SUPPRESS_WARNINGS["embed"] = True
            return
        
        # get obs shapes and parse obs
        self_obs = obs[i][:self_shape]
        blue_obs = obs[i][self_shape:(self_shape+blue_shape)]
        red_obs = obs[i][(self_shape+blue_shape):(self_shape+blue_shape+red_shape)]
        
        # agent_is_here
        _node = get_loc(self_obs, pos_obs_size)
        if _node == -1:
            print(ERROR_MSG("agent not found"))
        node_embeddings[i][_node][0] = 1
        
        # num_red_here
        for j in range(pos_obs_size):
            if red_obs[j]:
                node_embeddings[i][j][1] += 1
        
        # num_blue_here
        blue_positions = set([])
        for j in range(pos_obs_size):
            if blue_obs[j]:
                node_embeddings[i][j][2] += 1
                blue_positions.add(j)

        ## EXTRA EMBEDDINGS TO PROMOTE LEARNING ##
        # can_red_go_here_t
        for possible_next in map.g_acs.adj[_node+1]:
            node_embeddings[i][possible_next-1][3] = 1
        
        # can_blue_move_here_t
        if MOVE_DEGS["move_1deg_away"] == None:
            MOVE_DEGS["move_1deg_away"] = get_nodes_ndeg_away(map.g_acs.adj, 1)
        move_1deg_away = MOVE_DEGS["move_1deg_away"]
        for j in blue_positions:
            for possible_next in move_1deg_away[j+1]:
                node_embeddings[i][possible_next-1][4] = 1
        
        # add feature from some external "optimization", if desired
        if GRAPH_OBS_TOKEN["embed_opt"]:
            if OPT_SETTINGS["flanking"]: # check if node is a flanking position
                for j in range(pos_obs_size):
                    node_embeddings[i][j][5] = 1

    #node_embeddings[:,-1,:] = 0
    return node_embeddings.to(device)

# get location of an agent given one-hot positional encoding on graph (0-indexed)
def get_loc(one_hot_graph, graph_size, default=0):
    global SUPPRESS_WARNINGS
    for i in range(graph_size):
        if one_hot_graph[i]:
            return i
    if not SUPPRESS_WARNINGS["decode"]:
        print(f"test batch detected while decoding. agent not found. returning default={default} and suppressing this warning.")
        SUPPRESS_WARNINGS["decode"] = True
    return default


def get_nodes_ndeg_away(graph, n):
    """
    :param graph: networkx.adj adjacency dictionary
    :param n: number of degrees to search outwards
    :return dictionary mapping each node to a list of nodes n degrees away
    """
    result = {}
    for node in graph:
        result[node] = get_nodes_ndeg_from_s(graph, node, n)
    return result


def get_nodes_ndeg_from_s(graph, s, n):
    """
    collects the list of all nodes that are n degrees away from a source s on the given graph.
    :param graph: networkx.adj adjacency dictionary
    :param s: 1-indexed node starting location
    :param n: number of degrees to search outwards
    :return list of nodes that are n (or fewer) degrees from s
    """
    # run n iterations of bfs and collect node results
    visited = set([s])
    dq = deque([s])
    for i in range(n):
        if not dq:
            break
        node = dq.popleft()
        next_nodes = graph[node]
        for next_node in next_nodes:
            if next_node not in visited:
                visited.add(next_node)
                dq.append(next_node)
    return list(visited)


# load edge dictionary from a map (0-indexed)
def load_edge_dictionary(map_edges):
    """
    :param map_edges: edges from a graph from MapInfo. input should be 1-indexed MapInfo map_edge dictionary.
    :return the 0-indexed edge dictionary for quick lookups.
    """
    # create initial edge_array and TODO edge_to_action mappings
    edge_array = []
    for k, v in zip(map_edges.keys(), map_edges.values()):
        edge_array += [[k-1, vi-1] for vi in v.keys()]
    
    # create edge_dictionary
    edge_dictionary = {}
    for edge in edge_array:
        if edge[0] not in edge_dictionary: edge_dictionary[edge[0]] = set([])
        edge_dictionary[edge[0]].add(edge[1])
    
    return edge_dictionary


def get_cost_from_reward(reward):
    return 1/(reward + 1e-3) # takes care of div by 0


def get_probs_mask(agent_nodes, graph_size, edges_dict):
    node_exclude_list = np.array(list(range(graph_size)))
    mask = [np.delete(node_exclude_list, list(edges_dict[agent_node])+[agent_node]) for agent_node in agent_nodes]
    return mask


def count_model_params(model, print_model=False):
    num_params = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    if print_model: summary(model)
    print(f"{type(model)} using {num_params} #params")


def parse_config(model_config):
    hiddens = list(model_config.get("fcnet_hiddens", [])) + \
        list(model_config.get("post_fcnet_hiddens", []))
    activation = model_config.get("fcnet_activation")
    if not model_config.get("fcnet_hiddens", []):
        activation = model_config.get("post_fcnet_activation")
    no_final_linear = model_config.get("no_final_linear")
    vf_share_layers = model_config.get("vf_share_layers") # this is usually 0
    free_log_std = model_config.get("free_log_std") # skip worrying about log std
    return hiddens, activation, no_final_linear, vf_share_layers, free_log_std


def create_value_branch(obs_space, action_space, *, vf_share_layers=False, activation="relu", hiddens=[]):
    _value_branch_separate = None
    # create value network with equal number of hidden layers as policy net
    if not vf_share_layers:
        prev_vf_layer_size = int(np.product(obs_space.shape))
        vf_layers = []
        for size in hiddens:
            vf_layers.append(
                SlimFC(
                    in_size=prev_vf_layer_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(1.0)))
            prev_vf_layer_size = size
        _value_branch_separate = torch.nn.Sequential(*vf_layers)
    # layer which outputs 1 value
    #prev_layer_size = hiddens[-1] if self._value_branch_separate else self.map.get_graph_size()
    prev_layer_size = hiddens[-1] if _value_branch_separate else int(action_space.n)
    _value_branch = SlimFC(
        in_size=prev_layer_size,
        out_size=1,
        initializer=normc_initializer(0.01),
        activation_fn=None)
    return _value_branch, _value_branch_separate

"""
# junk #
        # can_blue_see_here_t,t+1,t+2
    #start = time.time()
    #print(time.time() - start, "embedding init time")
    #print("batch size is", batch_size)
    #start = time.time()
    #print(time.time() - start, "rest embed time")

L102

L115
        #    MOVE_DEGS["move_2deg_away"] = get_nodes_ndeg_away(map.g_acs.adj, 2)
        #    MOVE_DEGS["move_3deg_away"] = get_nodes_ndeg_away(map.g_acs.adj, 3)
        #move_2deg_away = MOVE_DEGS["move_2deg_away"]
        #move_3deg_away = MOVE_DEGS["move_3deg_away"]
            #for possible_next in move_2deg_away[j+1]:
            #    node_embeddings[i][possible_next-1][5] = 1
            #for possible_next in move_3deg_away[j+1]:
            #    node_embeddings[i][possible_next-1][6] = 1

            #VIEW_DEGS["view_1deg_away"] = get_nodes_ndeg_away(map.g_vis.adj, 1)
            #view_1deg_away = get_nodes_ndeg_away(map.g_vis.adj, 1)
            #view_2deg_away = get_nodes_ndeg_away(map.g_vis.adj, 2)
            #view_3deg_away = get_nodes_ndeg_away(map.g_vis.adj, 3)
            #move_1deg_away = get_nodes_ndeg_away(map.g_acs.adj, 1)
            #move_2deg_away = get_nodes_ndeg_away(map.g_acs.adj, 2)
            #move_3deg_away = get_nodes_ndeg_away(map.g_acs.adj, 3)
        #else:
        #view_1deg_away = VIEW_DEGS["view_1deg_away"]
        #view_2deg_away = VIEW_DEGS["view_2deg_away"]
        #view_3deg_away = VIEW_DEGS["view_3deg_away"]
        #move_2deg_away = MOVE_DEGS["move_2deg_away"]
        #move_3deg_away = MOVE_DEGS["move_3deg_away"]

L132

        #for j in range(pos_obs_size):
        #    if blue_obs[j]:
            for possible_next in view_1deg_away[j+1]:
                node_embeddings[i][possible_next-1][4] = 1
            for possible_next in view_2deg_away[j+1]:
                node_embeddings[i][possible_next-1][5] = 1
            for possible_next in view_3deg_away[j+1]:
                node_embeddings[i][possible_next-1][6] = 1
            


L144

            for possible_next in move_2deg_away[j+1]:
                node_embeddings[i][possible_next-1][8] = 1
            for possible_next in move_3deg_away[j+1]:
                node_embeddings[i][possible_next-1][9] = 1

"""

