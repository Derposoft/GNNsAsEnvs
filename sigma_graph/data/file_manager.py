import os
import re
from typing import Tuple

from .data_helper import get_node_name_from_pos_abs, get_node_pos_from_name_abs
from .graph.skirmish_graph import MapInfo, RouteInfo

# select which map to use
MAP_LOOKUP = {
    "S": "_27",
    "M": "_41",
    "L": "_73",
}

PATH_LOOKUP = {
    "saved": "sigma_graph/data/parsed/",
    "raw": "sigma_graph/data/raw/",
}

# assign a large int for no connectivity to replace "null" in graph generation pipeline
INDEX_INVAL = 911

# raw data files for parsing
RAW_MAP_DATA_LOOKUP = {
    "connectivity": "FCNNodeData_Figure8_27wp_NSWE.txt",
    "visibility": "visibility_nodes_Figure8_FOV120_NSWE.txt",
    "position": "coordinate_absolute.txt"
}

MAP_AGENT_DATA_LOOKUP = {
    "patrol_route": "wp_pat"
}

VISIBILITY_IS_360 = False


# lookup table for data types: prefixes of parsed data files for saving and loading
DATA_LOOKUP = {
    "connectivity": "graph_acs",
    "visibility": "graph_vis_360" if VISIBILITY_IS_360 else "graph_vis",
    "encoding": "info_dict_emb",
    "position": "info_dict_pos",
    "patrol_route": "info_list_pat"
}

def set_visibility(visibility):
    VISIBILITY_IS_360 = visibility;
    DATA_LOOKUP["visibility"] = "graph_vis_360" if VISIBILITY_IS_360 else "graph_vis"

def load_graph_files(env_path="./", map_lookup="S", route_lookup=["0"], is_pickle_graph=True):
    assert check_dir(env_path), "[GymEnv][Error] Invalid path for loading env data: \'{}\'".format(env_path)

    path_data = os.path.join(env_path, PATH_LOOKUP["saved"])
    assert check_dir(path_data), "[GymEnv][Error] Can not find data in: \'{}\'".format(path_data)

    map_id = MAP_LOOKUP[map_lookup]
    if is_pickle_graph:
        file_graph = 'pickle'
        file_data = 'pickle'
    else:
        file_graph = 'gexf'
        file_data = 'pkl'

    cur_map = MapInfo()
    graph_acs = find_file_in_dir(path_data, "{}{}.{}".format(DATA_LOOKUP["connectivity"], map_id, file_graph))
    graph_vis = find_file_in_dir(path_data, "{}{}.{}".format(DATA_LOOKUP["visibility"], map_id, file_graph))
    data_emb = find_file_in_dir(path_data, "{}{}.{}".format(DATA_LOOKUP["encoding"], map_id, file_data))
    data_pos = find_file_in_dir(path_data, "{}{}.{}".format(DATA_LOOKUP["position"], map_id, file_data))
    if is_pickle_graph:
        cur_map.load_graph_pickle(graph_acs, graph_vis, data_emb, data_pos)
    else:
        cur_map.load_graph_files(graph_acs, graph_vis, data_emb, data_pos)

    cur_pat = [RouteInfo()] * len(route_lookup)
    for idx in range(len(route_lookup)):
        data_pat = find_file_in_dir(path_data, "{}_{}{}.{}".format(DATA_LOOKUP["patrol_route"], route_lookup[idx],
                                                                   map_id, file_data))
        cur_pat[idx].load_route_pickle(data_pat)

    return cur_map, cur_pat


def generate_graph_files(env_path="./", map_lookup="S", route_lookup=['0'], is_pickle_graph=True, if_overwrite=True):
    assert check_dir(env_path), "[GymEnv][Error] Invalid path for graph data files: \'{}\'".format(env_path)
    path_file = os.path.join(env_path, PATH_LOOKUP["raw"])
    assert check_dir(path_file), "[GymEnv][Error] Can not find data in: {}".format(path_file)

    path_obj = os.path.join(env_path, PATH_LOOKUP["saved"])
    if not check_dir(path_obj):
        os.mkdir(path_obj)

    map_id = MAP_LOOKUP[map_lookup]
    # use default file storage for python3 objects
    if is_pickle_graph:
        file_graph = 'pickle'
        file_data = 'pickle'
    else:
        file_graph = 'gexf'
        file_data = 'pkl'

    # check exists of parsed files [option: overwrite existing files if the flag turns on]
    graph_acs, _acs = check_file_in_dir(path_obj, "{}{}.{}".format(DATA_LOOKUP["connectivity"], map_id, file_graph))
    graph_vis, _vis = check_file_in_dir(path_obj, "{}{}.{}".format(DATA_LOOKUP["visibility"], map_id, file_graph))
    obj_emb, _emb = check_file_in_dir(path_obj, "{}{}.{}".format(DATA_LOOKUP["encoding"], map_id, file_data))
    obj_pos, _pos = check_file_in_dir(path_obj, "{}{}.{}".format(DATA_LOOKUP["position"], map_id, file_data))
    obj_pat = [''] * len(route_lookup)
    for idx in range(len(route_lookup)):
        obj_pat[idx], _ = check_file_in_dir(path_obj, "{}_{}{}.{}".format(DATA_LOOKUP["patrol_route"],
                                                                          route_lookup[idx], map_id, file_data))

    if if_overwrite:
        if True in [_acs, _vis, _emb, _pos]:
            print("[GymEnv][Warning] This run will overwrite previous saved parsing results in \'{}\'".format(env_path))
        else:
            print("[GymEnv][Info] Start parsing raw data. Parsed data will be saved in \'{}\'".format(env_path))
    else:
        print("[GymEnv][Info] Start parsing raw data. Data will *NOT* save to files in this run.")

    # check exists of raw data files
    # find node connectivity file
    data_edge_acs = find_file_in_dir(path_file, RAW_MAP_DATA_LOOKUP["connectivity"])
    # find node visibility file
    data_edge_vis = find_file_in_dir(path_file, RAW_MAP_DATA_LOOKUP["visibility"])
    # find node absolute coordinate file
    data_node_pos = find_file_in_dir(path_file, RAW_MAP_DATA_LOOKUP["position"])
    # find patrol routes for blue agents
    data_route = [""] * len(route_lookup)
    for idx in range(len(route_lookup)):
        data_route[idx] = find_file_in_dir(path_file, "{}_{}.txt".format(MAP_AGENT_DATA_LOOKUP["patrol_route"],
                                                                         route_lookup[idx]))

    # generate a graph container instance
    cur_map = MapInfo()

    # IOs in node connectivity file
    file = open(data_edge_acs, 'r')
    lines = file.readlines()
    for line in lines:
        nodes = connection_line_parser(line)
        u_name = None
        for idx, node in enumerate(nodes):
            row, col = int(node[0]), int(node[1])
            if row == INDEX_INVAL:  # check placeholder for invalid 'null' actions [skip and continue]
                continue
            node_name = get_node_name_from_pos_abs((row, col))
            cur_map.add_node_acs(node_name)
            if idx:
                # add edges to all four target nodes
                cur_map.add_node_acs(node_name)
                # index number is the attribute for action lookup NSWE
                cur_map.add_edge_acs(u_name, node_name, idx)
            else:
                # the first node is the source node
                u_name = node_name

    # IOs in node visibility file
    file = open(data_edge_vis, 'r')
    lines = file.readlines()
    for line in lines:
        # get target node and source nodes in four directions
        u_node, v_list_N, v_list_S, v_list_W, v_list_E = visibility_fov_line_parser(line)
        u_name = get_node_name_from_pos_abs((int(u_node[0][0]), int(u_node[0][1])))
        if u_name in cur_map.n_name:
            node_dict = {1: v_list_N, 2: v_list_S, 3: v_list_W, 4: v_list_E}
            for idx in node_dict:
                for v_node in node_dict[idx]:
                    v_name = get_node_name_from_pos_abs((int(v_node[0]), int(v_node[1])))
                    if v_name in cur_map.n_name:
                        cur_map.add_edge_vis_fov(u_name, v_name, float(v_node[2]), idx)

    # IOs in node absolute coordinate file
    file = open(data_node_pos, 'r')
    lines = file.readlines()
    for line in lines:
        node, coors = coordinate_line_parser(line)
        node_name = get_node_name_from_pos_abs((int(node[0][0]), int(node[0][1])))
        if node_name in cur_map.n_name:
            cur_map.n_info[cur_map.n_name[node_name]] = (float(coors[0][0]), float(coors[0][2]))

    # generate path info
    cur_pat = []
    for idx in range(len(route_lookup)):
        idx_pat = RouteInfo()
        file = open(data_route[idx], 'r')
        lines = file.readlines()
        # set patrol route node list
        for line in lines:
            node_coor = patrol_route_line_parser(line)
            node_name = get_node_name_from_pos_abs((int(node_coor[0][0]), int(node_coor[0][1])))
            idx_pat.add_node_to_route(node_name)
        cur_pat.append(idx_pat)

    # save to file
    if if_overwrite:
        if is_pickle_graph:
            cur_map.save_graph_pickle(graph_acs, graph_vis, obj_emb, obj_pos)
        else:
            cur_map.save_graph_files(graph_acs, graph_vis, obj_emb, obj_pos)
        for idx in range(len(route_lookup)):
            cur_pat[idx].save_route_pickle(obj_pat[idx])
    return cur_map, cur_pat


def connection_line_parser(s):
    # change 'null' action in the raw data with a placeholder for better iterating and action matching
    s_acs = s.replace("null", "({},{})".format(INDEX_INVAL, INDEX_INVAL))
    s_nodes = re.findall(r"\((\d+),(\d+)\)", s_acs)
    # check if the list contains the source node and its neighbors in all four directions
    assert len(s_nodes) == 5, "[Parsing Error] Invalid node connections in line: \'{}\'".format(s_nodes)
    return s_nodes


def visibility_line_parser(s):
    s_s, s_t = s.split("\t")
    s_idx = re.findall(r"\((\d+),(\d+)\)", s_s)
    t_idx_dist = re.findall(r"\((\d+),(\d+),(\d+\.?\d*)\)", s_t)
    return s_idx, t_idx_dist


def visibility_fov_line_parser(s):
    s_nodes = re.split("\t", s)
    s_idx = re.findall(r"\((\d+),(\d+)\)", s_nodes[0])
    t1_idx_dist = re.findall(r"\((\d+),(\d+),(\d+\.?\d*)\)", s_nodes[1])
    t2_idx_dist = re.findall(r"\((\d+),(\d+),(\d+\.?\d*)\)", s_nodes[2])
    t3_idx_dist = re.findall(r"\((\d+),(\d+),(\d+\.?\d*)\)", s_nodes[3])
    t4_idx_dist = re.findall(r"\((\d+),(\d+),(\d+\.?\d*)\)", s_nodes[4])
    return s_idx, t1_idx_dist, t2_idx_dist, t3_idx_dist, t4_idx_dist


def coordinate_line_parser(s):
    idx, coor = s.split("\t")
    n_idx = re.findall(r"\((\d+),(\d+)\)", idx)
    n_coor = re.findall(r"\((\d+\.?\d*),\s(\d+\.?\d*),\s(\d+\.?\d*)\)", coor)
    return n_idx, n_coor


def patrol_route_line_parser(s):
    n_idx = re.findall(r"\[(\d+)\s*,\s*(\d+)\]", s)
    return n_idx


def find_file_in_dir(dir_name: str, file_name: str) -> str:
    path = os.path.join(dir_name, file_name)
    assert os.path.isfile(path), "[GymEnv][Error] Can not find file: {}.".format(path)
    return path


def check_file_in_dir(dir_name: str, file_name: str) -> Tuple[str, bool]:
    path = os.path.join(dir_name, file_name)
    return path, os.path.isfile(path)


def check_dir(dir_name: str) -> bool:
    return os.path.exists(dir_name)


# logger for step runs
def save_log_2_file(config, n_step, n_done, agents, prev_obs, actions, obs, rewards, dones=None):
    # ori_stdout = sys.stdout
    _log_path = os.path.join(config["root_path"], config["log_path"])
    if not check_dir(_log_path):
        os.makedirs(_log_path)
    file_path = os.path.join(_log_path, "{}done_{}.txt".format(config["log_prefix"], n_done))
    with open(file_path, 'a+') as f:
        # sys.stdout = f
        _buffer = "Step #{:2d} ".format(n_step)
        for _idx in range(len(agents)):
            _buffer += "| {} HP:{} node:{} dir:{} pos:{} ".format(agents[_idx][0], agents[_idx][3],
                                                                  agents[_idx][1][0], agents[_idx][1][1],
                                                                  get_node_pos_from_name_abs(agents[_idx][2]))
        _buffer += "| Actions:{} | Step rewards:{}".format(actions, rewards)
        # if config["save"] is True:
        print(_buffer, file=f)
        if config["log_verbose"]:
            # print("Done:{}".format(dones), file=f)
            _buffer_verbose = " | Obs_before:{} | Obs_after:{}".format(prev_obs, obs)
            # if config["save"] is True:
            print(_buffer_verbose, file=f)
        # sys.stdout = ori_stdout
    return True


# overview of episode rewards
def log_done_reward(config, n_done, rewards):
    _log_path = os.path.join(config["root_path"], config["log_path"])
    if not check_dir(_log_path):
        os.makedirs(_log_path)
    file_episode = os.path.join(_log_path, config["log_overview"])
    with open(file_episode, 'a+') as f:
        _episode = "Episode #{:2d} ends with episode_reward:{}".format(n_done, rewards)
        print(_episode, file=f)
    file_step = os.path.join(_log_path, "{}done_{}.txt".format(config["log_prefix"], n_done))
    with open(file_step, 'a+') as f:
        _step = "Episode rewards:{}".format(rewards)
        print(_step, file=f)
    return True


def generate_parsed_data_files():
    # relative path to project root
    _env_path = "./"
    _map_lookup = ["S"]  # ["S", "M", "L"]
    _route_lookup = ['0', '1']

    for _map in _map_lookup:
        generate_graph_files(env_path=_env_path, map_lookup=_map, route_lookup=_route_lookup,
                             is_pickle_graph=True, if_overwrite=True)


if __name__ == "__main__":
    generate_parsed_data_files()
