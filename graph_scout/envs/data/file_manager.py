import os
import re

from graph_scout.envs.data.terrain_graph import MapInfo
import graph_scout.envs.data.file_lookup as fc


def load_graph_files(env_path="./", map_lookup="Std"):
    assert check_dir(env_path), "[GSMEnv][File] Invalid path for loading env data: {}".format(env_path)

    path_data = os.path.join(env_path, fc.PATH_LOOKUP["file_o"])
    assert check_dir(path_data), "[GSMEnv][File] Can not find data in: {}".format(path_data)

    map_id = fc.MAP_LOOKUP[map_lookup]
    graph_move = find_file_in_dir(path_data, "{}{}.pickle".format(fc.DATA_LOOKUP["d_connectivity"], map_id))
    graph_view = find_file_in_dir(path_data, "{}{}.pickle".format(fc.DATA_LOOKUP["d_visibility"], map_id))
    data_table = find_file_in_dir(path_data, "{}{}.pickle".format(fc.DATA_LOOKUP["d_mapping"], map_id))
    data_coord = find_file_in_dir(path_data, "{}{}.pickle".format(fc.DATA_LOOKUP["d_coordinates"], map_id))

    cur_map = MapInfo()
    cur_map.load_graph_pickle(graph_move, graph_view, data_table, data_coord)

    return cur_map


def generate_graph_files(env_path="./", map_lookup="Std", if_overwrite=True):
    assert check_dir(env_path), "[GSMEnv][File] Invalid path for graph data files: \'{}\'".format(env_path)
    path_file = os.path.join(env_path, fc.PATH_LOOKUP["file_i"])
    assert check_dir(path_file), "[GSMEnv][File] Can not find data in: {}".format(path_file)

    path_obj = os.path.join(env_path, fc.PATH_LOOKUP["file_o"])
    if not check_dir(path_obj):
        os.mkdir(path_obj)

    map_id = fc.MAP_LOOKUP[map_lookup]

    # check exists of parsed files [option: overwrite existing files if the flag turns on]
    graph_move, _move = check_file_in_dir(path_obj, "{}{}.pickle".format(fc.DATA_LOOKUP["d_connectivity"], map_id))
    graph_view, _view = check_file_in_dir(path_obj, "{}{}.pickle".format(fc.DATA_LOOKUP["d_visibility"], map_id))
    obj_map, _map = check_file_in_dir(path_obj, "{}{}.pickle".format(fc.DATA_LOOKUP["d_mapping"], map_id))
    obj_pos, _pos = check_file_in_dir(path_obj, "{}{}.pickle".format(fc.DATA_LOOKUP["d_coordinates"], map_id))

    if if_overwrite:
        if True in [_move, _view, _map, _pos]:
            print("[GSMEnv][Info] Overwrite previous saved parsing data in \'{}\'".format(env_path))
        else:
            print("[GSMEnv][Info] Start parsing raw data. Parsed data will be saved in \'{}\'".format(env_path))
    else:
        print("[GSMEnv][Info] Start parsing raw data. Will NOT save or overwrite files. <online mode>")

    # check existence of raw data files

    # check node connectivity file
    data_raw_move = find_file_in_dir(path_file, fc.RAW_DATA_LOOKUP["r_connectivity"])
    # check node absolute coordinate file
    data_raw_coor = find_file_in_dir(path_file, fc.RAW_DATA_LOOKUP["r_coordinates"])
    # check visibility & probability files
    data_raw_view = [find_file_in_dir(path_file, fc.RAW_DATA_LOOKUP["r_visibility"][_file]) for _file in
                     fc.RAW_DATA_LOOKUP["r_visibility"]]

    # preprocessing & utilities for raw data conventions
    # from graph_scout.envs.data.node_coor_mapping import dict_node_id_pos
    from node_coor_mapping import dict_node_id_pos
    from copy import deepcopy
    dict_table = dict_node_id_pos

    list_n_id = list(dict_table.keys())
    list_n_loc = list(dict_table.values())

    def get_id_from_2D_coord(_row, _col):
        return list_n_id[list_n_loc.index((_row, _col))]

    # generate a graph container instance
    cur_map = MapInfo()
    cur_map.n_table = deepcopy(dict_table)
    cur_map.add_node_init_list(list_n_id)

    # parse data in node connectivity file
    with open(data_raw_move, 'r') as file:
        lines = file.readlines()
        for line in lines:
            list_pos = connection_line_parser(line)
            for index, node in enumerate(list_pos):
                row, col = int(node[0]), int(node[1])
                # check placeholder (0,0) for invalid 'null' actions [skip and continue]
                if row == 0 and col == 0:
                    continue
                n_id = get_id_from_2D_coord(row, col)
                if index:
                    # directed edge source->target with the attribute for action lookup 'NSWE->1234'
                    cur_map.add_edge_Gmove(u_id, n_id, index)
                else:
                    # the first node (index == 0) is the source node in each line
                    u_id = n_id

    # parse data in node absolute coordinate file
    with open(data_raw_coor, 'r') as file:
        lines = file.readlines()
        for line in lines:
            s_pos, s_coord = coordinate_line_parser(line)
            n_id = get_id_from_2D_coord(int(s_pos[0][0]), int(s_pos[0][1]))
            if n_id in cur_map.n_table:
                # store X & Z coordinates for plotting
                cur_map.n_coord[n_id] = (float(s_coord[0][0]), float(s_coord[0][2]))
                # store elevation info for interacting calculations
                cur_map.g_move.nodes[n_id]["height"] = float(s_coord[0][1])
            else:
                raise ValueError("[GSMEnv][Data] Invalid node coordinates.")

    # parse data in all visibility & probability files
    for f_index, data_view in enumerate(data_raw_view):
        with open(data_view, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # get target node and source nodes in four directions
                u_node, v_list_N, v_list_S, v_list_W, v_list_E = visibility_fov_line_parser(line)
                u_id = get_id_from_2D_coord(int(u_node[0][0]), int(u_node[0][1]))
                node_dict = {1: v_list_N, 2: v_list_S, 3: v_list_W, 4: v_list_E}
                for d_index in node_dict:
                    for v_node in node_dict[d_index]:
                        v_id = get_id_from_2D_coord(int(v_node[0]), int(v_node[1]))
                        # attrs: direction, posture, probability, distance
                        cur_map.add_edge_Gview_FOV(u_id, v_id, d_index, f_index, float(v_node[-1]), float(v_node[2]))

    # save to file
    if if_overwrite:
        cur_map.save_graph_pickle(graph_move, graph_view, obj_map, obj_pos)
        # [TBD] might need to modify: API Deprecated since nx version 2.6.
    return cur_map


def connection_line_parser(s):
    # replace 'null' action by a virtual node placeholder (0,0) in the raw data for better action iterating and matching
    s_acts = s.replace("null", "(0,0)")
    s_nodes = re.findall(r"\((\d+),(\d+)\)", s_acts)
    # check if the list contains the source node and its neighbors in all four directions
    assert len(s_nodes) == 5, f"[GSMEnv][Parser] Invalid node connections in line: \'{s_nodes}\'"
    return s_nodes


def coordinate_line_parser(s):
    s_pos, s_coord = s.split("\t")
    n_pos = re.findall(r"\((\d+),(\d+)\)", s_pos)
    n_coord = re.findall(r"\((\d+\.?\d*),\s(\d+\.?\d*),\s(\d+\.?\d*)\)", s_coord)
    return n_pos, n_coord


def visibility_line_parser(s):
    s_s, s_t = s.split("\t")
    s_node = re.findall(r"\((\d+),(\d+)\)", s_s)
    d_nodes = visual_prob_findall(s_t)
    return s_node, d_nodes


def visibility_fov_line_parser(s):
    s_nodes = re.split("\t", s)
    s_node = re.findall(r"\((\d+),(\d+)\)", s_nodes[0])
    # generate lists for all looking directions
    d_list_N = visual_prob_findall(s_nodes[1])
    d_list_S = visual_prob_findall(s_nodes[2])
    d_list_W = visual_prob_findall(s_nodes[3])
    d_list_E = visual_prob_findall(s_nodes[4])
    return s_node, d_list_N, d_list_S, d_list_W, d_list_E


# get a list of all adjacency nodes with 'dist' and 'prob' strings
def visual_prob_findall(s):
    return re.findall(r"\((\d+),(\d+),(\d+\.?\d*)\)\|(\d)\|\D*\|([0-9.eE\-]+);", s)


# verify probability values
def visual_prob_check_num(s_prob):
    return re.search(r"(\d)(\.\d*)?([eE]-?(\d+))?", s_prob)


# omit body parts tokens: ['tuple', 'prob']
def visual_prob_elem_parser(s):
    elem_list = re.split(r';', s)
    e_list = []
    for elem in elem_list:
        e_list.append(re.split(r'\|\d\|\D*\|', elem))
    return e_list


def find_file_in_dir(dir_name: str, file_name: str) -> str:
    path = os.path.join(dir_name, file_name)
    assert os.path.isfile(path), f"[Parser][Error] Can not find file: {path}."
    return path


def check_file_in_dir(dir_name: str, file_name: str):
    path = os.path.join(dir_name, file_name)
    return path, os.path.isfile(path)


def check_dir(dir_name: str) -> bool:
    return os.path.exists(dir_name)


def generate_parsed_data_files():
    # relative path to project root
    _env_path = "./"
    _map_lookup = ["Std"]
    for _map in _map_lookup:
        generate_graph_files(env_path=_env_path, map_lookup=_map, if_overwrite=True)


if __name__ == "__main__":
    generate_parsed_data_files()