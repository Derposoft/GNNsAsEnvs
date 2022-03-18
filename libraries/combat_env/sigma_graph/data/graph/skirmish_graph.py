import networkx as nx
import pickle


class MapInfo:
    # skirmish simulation map with 4-way connected waypoints and FOV-based visibility constraints
    def __init__(self):
        self.g_acs = nx.DiGraph(method="get_action")  # {node type: idx}, {node label: encoding}
        self.g_vis = nx.MultiDiGraph(method="get_distance")  # {node type: idx}, {edge label: distance & directions}
        self.n_name = dict()  # lookup table for nodes {name: idx}
        self.n_info = dict()  # {idx: (x,z)} node X and Z absolute coordinates for 2D graph drawing
        self.counter = 0

    def add_node_acs(self, node_name) -> bool:
        # add node to action graph and update node dict
        if node_name in self.n_name:
            return True
        self.counter += 1
        idx = self.counter
        self.n_name[node_name] = idx
        self.g_acs.add_node(idx, code=node_name)
        return False

    def add_node_vis_by_name(self, node_name, attrs=None) -> bool:
        # add node to visual graph if node is in dict
        if node_name in self.n_name:
            self.g_vis.add_node(self.n_name[node_name])
            # add node init attrs
            # self.g_vis.add_node(self.n_name[node_name], attrs)
            return False
        return True

    def add_node_vis_by_index(self, idx, **args) -> bool:
        # add node to visual graph if node is in dict
        if idx in range(1, len(self.n_name) + 1):
            self.g_vis.add_node(idx, **args)
        return False

    def add_edge_acs(self, u_name, v_name, attr) -> bool:
        # check node existence first, avoid to add new nodes here
        if u_name in self.n_name and v_name in self.n_name:
            self.g_acs.add_edge(self.n_name[u_name], self.n_name[v_name], action=attr)
            return False
        return True

    def add_edge_vis_fov(self, u_name, v_name, attr_dist, attr_type) -> bool:
        # check node first: avoid to add new node that is not in the lookup dict
        if u_name in self.n_name and v_name in self.n_name:
            u_node, v_node = self.n_name[u_name], self.n_name[v_name]
            # set the distance attribute to the first edge if there are parallel edges
            if self.g_vis.has_edge(u_node, v_node):
                self.g_vis.add_edge(u_node, v_node, type=attr_type)
            else:
                self.g_vis.add_edge(u_node, v_node, type=attr_type)
                self.g_vis[u_node][v_node][0]['dist'] = attr_dist
            return False
        return True

    def reset(self):
        # if not (nx.is_frozen(self.g_acs) and nx.is_frozen(self.g_vis)):
        #     self.g_acs.clear()
        #     self.g_vis.clear()
        self.g_acs = nx.DiGraph(method="get_action")
        self.g_vis = nx.MultiDiGraph(method="get_distance")
        self.n_name = dict()
        self.n_info = dict()
        self.counter = 0

    def set_draw_attrs(self, node_name, x, z):
        # store 'pos' absolute coordinates attribute for drawing
        if node_name in self.n_name:
            idx = self.n_name[node_name]
            self.n_info[idx] = (x, z)
        return False

    def get_graph_size(self):
        return self.counter

    def get_graph_size_verbose(self):
        return self.counter, len(self.n_name), len(self.n_info), len(self.g_acs), len(self.g_vis)

    def get_index_by_name(self, node_name):
        return self.n_name[node_name]

    def get_name_by_index(self, index):
        return self.g_acs.nodes[index]["code"]

    def get_edge_attr_acs_by_idx(self, u_idx, v_idx):
        # no edge check for faster access
        return self.g_acs[u_idx][v_idx]["action"]

    def get_edge_attr_acs_by_name(self, u_name, v_name):
        # no edge check for faster access
        return self.g_acs[self.n_name[u_name]][self.n_name[v_name]]["action"]

    def get_edge_attr_vis_by_idx(self, u_idx, v_idx):
        # no edge check for faster access
        return self.g_vis[u_idx][v_idx][0]["dist"]

    def get_edge_attr_vis_fov_by_idx(self, u_idx, v_idx, u_dir):
        # check all parallel edges(u, v), return the distance value (>0) if the looking direction is valid
        dirs = [self.g_vis[u_idx][v_idx][idx]['type'] for idx in self.g_vis[u_idx][v_idx]]
        # return the distance value or a -1 indicator
        return self.g_vis[u_idx][v_idx][0]["dist"] if (u_dir in dirs) else -1

    def get_actions_by_node(self, node_name):
        s_idx = self.get_index_by_name(node_name)
        ts_idx = list(nx.neighbors(self.g_acs, s_idx))
        # send valid actions in ACTION_LOOKUP
        return [self.get_edge_attr_acs_by_idx(s_idx, t_idx) for t_idx in ts_idx]

    def get_all_states_by_node(self, node_name):
        s_idx = self.get_index_by_name(node_name)
        ts_idx = list(nx.neighbors(self.g_acs, s_idx))
        # send the whole 1st order subgraph (current_index, list_of_neighbor_index, list_of_action_nums)
        return s_idx, ts_idx, [self.get_edge_attr_acs_by_idx(s_idx, t_idx) for t_idx in ts_idx]

    def draw_graphs(self):
        img_acs = nx.draw_networkx(self.g_acs, pos=self.n_info, arrows=True)
        img_vis = nx.draw_networkx(self.g_vis, pos=self.n_info, arrows=True, with_labels=True)
        return img_acs, img_vis

    def get_draw_info_graph_acs(self):
        # get node positions and labels for connectivity graph visualization
        g_pos = self.n_info
        g_node_labels = nx.get_node_attributes(self.g_acs, "code")
        return g_pos, g_node_labels

    def get_draw_info_graph_vis(self):
        # get node positions and labels for visibility graph visualization
        # connectivity graph and visibility graph have the same node orders
        g_edge_labels = nx.get_edge_attributes(self.g_vis, "dist")
        return g_edge_labels

    def save_plots_to_file(self, f_acs, f_vis) -> bool:
        import matplotlib.pyplot as plt
        from datetime import datetime
        ts = datetime.now()
        timestamp = "_{}-{}-{}-{}".format(ts.month, ts.day, ts.hour, ts.minute)
        # save the plot of connectivity graph
        plt.figure()
        plt.axis('off')
        nx.draw_networkx(self.g_acs, self.n_info, arrows=True)
        # directed edges overlap so just one directional edge label can be seen in image
        # g_edge_labels = nx.get_edge_attributes(self.g_acs, "action")
        # nx.draw_networkx_edge_labels(self.g_acs, self.n_info, edge_labels=g_edge_labels)
        plt.savefig("{}{}.png".format(f_acs, timestamp))
        # save the plot of visibility graph
        plt.figure()
        plt.axis('off')
        nx.draw_networkx(self.g_vis, pos=self.n_info, with_labels=True)
        # edge labels are too dense which make it hard to get a clear view if plots all
        # g_edge_labels = nx.get_edge_attributes(self.g_vis, "dist")
        # nx.draw_networkx_edge_labels(self.g_vis, pos=self.n_info, edge_labels=g_edge_labels)
        plt.savefig("{}{}.png".format(f_vis, timestamp))

    def save_graph_files(self, f_acs, f_vis, f_name, f_info) -> bool:
        # [!!!] <NX gexf read/write xml bug> IOs change node name type from Int (i.e. 11) ot Str (i.e. '11')
        # cause problems in drawing
        nx.write_gexf(self.g_acs, f_acs)
        nx.write_gexf(self.g_vis, f_vis)
        with open(f_name, 'wb+') as file:
            pickle.dump(self.n_name, file, pickle.HIGHEST_PROTOCOL)
        with open(f_info, 'wb+') as file:
            pickle.dump(self.n_info, file, pickle.HIGHEST_PROTOCOL)

    def load_graph_files(self, f_acs, f_vis, f_name, f_info) -> bool:
        self.g_acs = nx.read_gexf(f_acs)
        self.g_vis = nx.read_gexf(f_vis)
        with open(f_name, 'rb') as file:
            self.n_name = pickle.load(file)
        with open(f_info, 'rb') as file:
            self.n_info = pickle.load(file)
        # check length
        if len(self.n_name) == len(self.n_info) and len(self.n_name) == len(self.g_acs):
            self.counter = len(self.n_name)
            # nx.freeze(self.g_acs)
            # nx.freeze(self.g_vis)
            return False
        else:
            print("[GymEnv][IO] Fatal error while loading graph xml files..")
            return True

    def save_graph_pickle(self, f_acs, f_vis, f_name, f_info):
        # all data saved in the pickle fashion
        nx.write_gpickle(self.g_acs, f_acs)
        nx.write_gpickle(self.g_vis, f_vis)
        with open(f_name, 'wb+') as file:
            pickle.dump(self.n_name, file, pickle.HIGHEST_PROTOCOL)
        with open(f_info, 'wb+') as file:
            pickle.dump(self.n_info, file, pickle.HIGHEST_PROTOCOL)

    def load_graph_pickle(self, f_acs, f_vis, f_name, f_info) -> bool:
        self.g_acs = nx.read_gpickle(f_acs)
        self.g_vis = nx.read_gpickle(f_vis)
        with open(f_name, 'rb') as file:
            self.n_name = pickle.load(file)
        with open(f_info, 'rb') as file:
            self.n_info = pickle.load(file)
        # check length
        if len(self.n_name) == len(self.n_info) and len(self.n_name) == len(self.g_acs):
            self.counter = len(self.n_name)
            return False
        else:
            print("[GymEnv][IO] Fatal error while loading graph pickle files..")
            return True


class ActGraph(nx.DiGraph):
    def __init__(self):
        super(nx.DiGraph, self).__init__()
        self.node_list = list()
        self.edge_relational = list()
        self.route_counter = 0
        self.route_list = list()        # type=RouteInfo() for multiple patrol routes


class VisGraph(nx.Graph):
    def __init__(self):
        super(nx.MultiDiGraphGraph, self).__init__()
        self.edge_list = list()
        self.weights = list()
        self.positions = list()


class RouteInfo:
    def __init__(self):
        # load from files
        self.list_code = list()  # encodings of the nodes
        # generate in runtime
        self.list_node = list()  # nodes in the patrol route
        self.list_move = list()     # moving direction in the current step
        self.list_next = list()     # moving direction in the next step for fast retrieval

    def save_route_pickle(self, f_route):
        with open(f_route, 'wb+') as file:
            pickle.dump(self.list_code, file, pickle.HIGHEST_PROTOCOL)

    def load_route_pickle(self, f_route):
        with open(f_route, 'rb') as file:
            self.list_code = pickle.load(file)

    def save_route(self, f_route):
        with open(f_route + '.pkl', 'wb+') as file:
            pickle.dump(self.list_code, file, pickle.HIGHEST_PROTOCOL)

    def load_route(self, f_route):
        with open(f_route + '.pkl', 'rb') as file:
            self.list_code = pickle.load(file)

    def add_node_to_route(self, node):
        self.list_code.append(node)

    def reset(self):
        pass

    def generate_path_graph(self):
        # self.g_pat = nx.path_graph(self.node_list)
        # self.g_pat.add_edge(self.get_node_by_index(-1), self.get_node_by_index(0))    # close loop
        pass

    def get_node_by_index(self, index: int) -> str:
        return self.list_code[index]

    def get_next_move_by_index(self, index: int):
        return self.list_next[index]

    def get_location_by_index(self, index: int):
        return self.list_node[index], self.list_code[index], self.list_move[index]

    def get_index_by_code(self, code: str) -> int:
        return self.list_code.index(code)

    def get_route_length(self) -> int:
        return len(self.list_code)
