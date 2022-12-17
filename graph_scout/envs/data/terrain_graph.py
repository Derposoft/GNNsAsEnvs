import networkx as nx
import pickle


class MapInfo:
    # terrain graphs with up to 4-way connected grid-like waypoints + FOV-based visibilities & damage probabilities
    def __init__(self):
        # {node: index_id, node_label: height, edge_label: direction}
        self.g_move = nx.DiGraph(method="get_action")
        # {node: index_id, edge_labels: (distance), direction, posture & probabilities}
        self.g_view = nx.MultiDiGraph(method="get_distance")
        # {n_id: (row, col)} relative 2D coordinates
        self.n_table = dict()
        # {n_id: (X, Z)} absolute 3D coordinates for visualization
        self.n_coord = dict()
        self.counter = 0

    def add_node_init_list(self, list_n_id) -> bool:
        # if not self.counter:
        #     return True
        
        # fast init without sanity checks
        self.g_move.add_nodes_from(list_n_id)
        self.g_view.add_nodes_from(list_n_id)
        self.counter = len(list_n_id)
        return False

    def add_node_Gmove_single(self, n_id, **dict_attrs) -> bool:
        # add node to action graph with attrs
        if n_id in self.n_table:
            return True
        self.g_move.add_node(n_id, **dict_attrs)
        return False

    def add_node_Gview_single(self, n_id, **dict_attrs) -> bool:
        # add node to visual graph with attrs
        if n_id in self.n_table:
            return True
        self.g_view.add_node(n_id, **dict_attrs)
        return False

    def add_edge_Gmove(self, u_id, v_id, attr_action) -> bool:
        # check node existence first
        if u_id in self.n_table and v_id in self.n_table:
            # attr value: action lookup indexing number
            self.g_move.add_edge(u_id, v_id, action=attr_action)
            return False
        else:
            raise KeyError("[GSMEnv][Graph] Invalid node index.")
            # return True

    def add_edge_Gview_FOV(self, u_id, v_id, attr_dir, attr_pos, attr_prob, attr_dist) -> bool:
        # check node existence first
        if u_id in self.n_table and v_id in self.n_table:
            # set the distance attribute to the first edge if there are parallel edges
            if self.g_view.has_edge(u_id, v_id):
                self.g_view.add_edge(u_id, v_id, dir=attr_dir, posture=attr_pos, prob=attr_prob)
            else:
                # only edge[u][v][0] has the distance attr
                self.g_view.add_edge(u_id, v_id, dir=attr_dir, posture=attr_pos, prob=attr_prob, dist=attr_dist)
            return False
        else:
            raise KeyError("[GSMEnv][Graph] Invalid node index.")
            # return True

    def reset(self):
        # if not (nx.is_frozen(self.g_move) and nx.is_frozen(self.g_view)):
        #     self.g_move.clear()
        #     self.g_view.clear()
        self.g_move = nx.DiGraph(method="get_action")
        self.g_view = nx.MultiDiGraph(method="get_distance")
        self.n_table = dict()
        self.n_coord = dict()
        self.counter = 0

    def set_draw_attrs(self, n_id, coord):
        # store absolute coordinates as attributes for drawing graphs
        if n_id in self.n_table:
            self.n_coord[n_id] = coord
        else:
            raise KeyError("[GSMEnv][Graph] Invalid node index.")
        return False

    def get_graph_size(self):
        return self.counter

    def get_graph_size_verbose(self):
        return self.counter, len(self.g_move), len(self.g_view), len(self.n_table), len(self.n_coord)

    def get_Gmove_edge_attr(self, u_id, v_id):
        # no edge check for fast accessing: value -> action index
        return self.g_move[u_id][v_id]["action"]

    def get_Gview_edge_attr_dict(self, u_id, v_id, e_id):
        # no edge sanity check: (node_s, node_t, edge_index) -> dict {all attrs}
        return self.g_view[u_id][v_id][e_id]

    def get_Gview_edge_attr_pos(self, u_id, v_id, e_id=0):
        return self.g_view[u_id][v_id][e_id]["posture"]

    def get_Gview_edge_attr_dir(self, u_id, v_id, e_id=0):
        return self.g_view[u_id][v_id][e_id]["dir"]

    def get_Gview_edge_attr_prob(self, u_id, v_id, e_id):
        return self.g_view[u_id][v_id][e_id]["prob"]

    def get_Gview_edge_attr_dist(self, u_id, v_id):
        return self.g_view[u_id][v_id][0]["dist"]

    def get_Gview_prob_by_dir_pos(self, u_id, v_id, u_dir, pos_u_v):
        # check all parallel edges(u, v), return edge_id & prob if it has edge (dir, pos)
        edge_view = self.g_view[u_id][v_id]
        _edge_index = [True if (edge_view[_id]['dir'] == u_dir) and (edge_view[_id]['posture'] == pos_u_v) else False for _id in edge_view]
        # return the prob value (-1 -> none indicator)
        if any(_edge_index):
            _e = _edge_index.index(True)
            return _e, edge_view[_e]["prob"]
        return -1, 0

    def get_Gview_neighbor_by_dir_pos(self, u_id, u_dir, pos_u_v):
        # return all valid nodes
        u_adj = self.g_view[u_id]
        list_adj = {}
        for _node in u_adj:
            for _edge in u_adj[_node]:
                if u_adj[_node][_edge]['dir'] == u_dir and u_adj[_node][_edge]['posture'] == pos_u_v:
                    list_adj[_node] = _edge
        return list_adj

    def get_Gmove_all_action(self, n_id):
        list_t_id = list(nx.neighbors(self.g_move, n_id))
        # get all valid action tokens from 'ACTION_LOOKUP' table
        return [self.get_Gmove_edge_attr(n_id, t_id) for t_id in list_t_id]

    def get_Gmove_action_node_dict(self, n_id):
        adj_id = list(nx.neighbors(self.g_move, n_id))
        # send the whole 1st order subgraph (current_index, list_of_neighbor_index, list_of_action_nums)
        dict_dir_target = dict()
        for t_id in adj_id:
            dict_dir_target[self.get_Gmove_edge_attr(n_id, t_id)] = t_id
        return dict_dir_target

    # get the partial path to the target while keeping distance (first visible node)
    def get_Gmove_path(self, n_src, n_tar, dist_neighbor=2):
        full_path = self.get_Gmove_shortest_path(n_src, n_tar)
        _len = len(full_path)
        if _len:
            if _len < dist_neighbor + 1:
                return full_path
            path = full_path[:-dist_neighbor]
            tail = full_path[-dist_neighbor:-1]
            for _tar in tail:
                if not self.g_move.has_edge(_tar, n_tar):
                    path.append(_tar)
            return path
        else:
            raise KeyError("[GSMEnv][Graph] Invalid source target pair")

    # get the shortest path by a pair of node_id
    def get_Gmove_shortest_path(self, n_src, n_tar):
        return nx.shortest_path(self.g_move, source=n_src, target=n_tar)

    def get_draw_attr_3D(self):
        # get node positions and labels for connectivity graph visualization
        label_coord = self.n_coord
        label_height = nx.get_node_attributes(self.g_move, "height")
        return label_coord, label_height

    def get_draw_attr_2D(self):
        return self.n_coord

    def get_draw_attr_Gview(self):
        # get node positions and labels for visibility graph visualization
        g_edge_labels = nx.get_edge_attributes(self.g_view, "dist")
        return g_edge_labels

    def save_graph_pickle(self, f_move, f_view, f_table, f_coord):
        # all data saved in the pickle fashion
        nx.write_gpickle(self.g_move, f_move)
        nx.write_gpickle(self.g_view, f_view)
        with open(f_table, 'wb') as file:
            pickle.dump(self.n_table, file, pickle.HIGHEST_PROTOCOL)
        with open(f_coord, 'wb') as file:
            pickle.dump(self.n_coord, file, pickle.HIGHEST_PROTOCOL)

    def load_graph_pickle(self, f_move, f_view, f_table, f_coord) -> bool:
        self.g_move = nx.read_gpickle(f_move)
        self.g_view = nx.read_gpickle(f_view)
        with open(f_table, 'rb') as file:
            self.n_table = pickle.load(file)
        with open(f_coord, 'rb') as file:
            self.n_coord = pickle.load(file)

        # check length
        n_count = len(self.n_table)
        if n_count == len(self.n_coord) and n_count == len(self.g_move):
            self.counter = n_count
            return False
        else:
            raise KeyError("[GSMEnv][Graph] Fatal error in loading pickle files. Please check raw data and try again.")
            # return True