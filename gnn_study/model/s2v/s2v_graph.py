import ctypes

class S2VGraph(object):
    #def __init__(self, handle, smiles, pce):
    def __init__(self, handle):
        self.handle = ctypes.c_void_p(handle)
        self.num_nodes = MOLLIB.lib.NumNodes(self.handle)
        self.num_edges = MOLLIB.lib.NumEdges(self.handle)
        # self.edge_pairs = np.ctypeslib.as_array(MOLLIB.lib.EdgeList(self.handle), shape=( self.num_edges * 2, ))
        self.edge_pairs = ctypes.c_void_p(MOLLIB.lib.EdgeList(self.handle))