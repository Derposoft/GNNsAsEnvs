# select which map to use
MAP_LOOKUP = {
    "Std": "_116",
}


PATH_LOOKUP = {
    "file_i": "graph_scout/envs/data/raw/",
    "file_o": "graph_scout/envs/data/parsed/"
}


# raw data files for parsing
RAW_DATA_LOOKUP = {
    "r_connectivity": "connectivity_NSWE_116.txt",
    "r_coordinates": "node_lookuptable_116.txt",
    "r_visibility": {0: "visibility_source-stand_target-stand_FOV180.txt",
                     1: "visibility_source-stand_target-prone_FOV180.txt",
                     2: "visibility_source-prone_target-stand_FOV180.txt",
                     3: "visibility_source-prone_target-prone_FOV180.txt"}
}
# pairwise posture mapping dict_pos_pair = {0:S_S, 1:S_P, 2:P_S, 3:P_P}


# lookup table for data types: prefixes of parsed data files for saving and loading
DATA_LOOKUP = {
    "d_connectivity": "GSM_graph_move",
    "d_visibility": "GSM_graph_view",
    "d_coordinates": "GSM_dict_coord",
    "d_mapping": "GSM_dict_table",
}