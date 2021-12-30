from typing import List
import ctypes
import torch

class S2VGraph(object):
    #def __init__(self, handle, smiles, pce):
    def __init__(self, num_nodes, num_edges, edge_pairs):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        # self.edge_pairs = np.ctypeslib.as_array(MOLLIB.lib.EdgeList(self.handle), shape=( self.num_edges * 2, ))
        self.edge_pairs = edge_pairs


def PrepareFeatureLabel(self, graphs: List[S2VGraph]):
    c_list = (ctypes.c_void_p * len(graphs))()
    total_num_nodes = 0
    total_num_edges = 0
    for i in range(len(graphs)):
        c_list[i] = graphs[i].handle
        total_num_nodes += graphs[i].num_nodes
        total_num_edges += graphs[i].num_edges

    torch_node_feat = torch.zeros(total_num_nodes, self.num_node_feats)
    torch_edge_feat = torch.zeros(total_num_edges * 2, self.num_edge_feats)
    torch_label = torch.zeros(len(graphs), 1)

    node_feat = torch_node_feat.numpy()
    edge_feat = torch_edge_feat.numpy()    
    label = torch_label.numpy()

    self.lib.PrepareBatchFeature(len(graphs), ctypes.cast(c_list, ctypes.c_void_p),
                                ctypes.c_void_p(node_feat.ctypes.data), 
                                ctypes.c_void_p(edge_feat.ctypes.data))

    for i in range(len(graphs)):
        label[i] = graphs[i].pce

    return torch_node_feat, torch_edge_feat, torch_label

def PrepareBatchFeature(num_graphs: int):#), void** g_list, Dtype* node_input, Dtype* edge_input):
    pass

'''

int PrepareBatchFeature(const int num_graphs, void** g_list, Dtype* node_input, Dtype* edge_input)
{    
    unsigned edge_cnt = 0, node_cnt = 0;
    
    for (int i = 0; i < num_graphs; ++i)
    {
        MolGraph* g = static_cast<MolGraph*>(g_list[i]);
		node_cnt += g->num_nodes;
		edge_cnt += g->num_edges;
    }
    
    Dtype* ptr = node_input;
    for (int i = 0; i < num_graphs; ++i)
    {
        MolGraph* g = static_cast<MolGraph*>(g_list[i]);

		for (int j = 0; j < g->num_nodes; ++j)
		{
			MolFeat::ParseAtomFeat(ptr, g->node_feat_at(j));
			ptr += MolFeat::nodefeat_dim;
		}
    }

	ptr = edge_input;
	for (int i = 0; i < num_graphs; ++i)
	{
        MolGraph* g = static_cast<MolGraph*>(g_list[i]);
		for (int j = 0; j < g->num_edges * 2; j += 2)
		{
			// two directions have the same feature
			MolFeat::ParseEdgeFeat(ptr, g->edge_feat_at(j / 2));
			ptr += MolFeat::edgefeat_dim;
			MolFeat::ParseEdgeFeat(ptr, g->edge_feat_at(j / 2));
			ptr += MolFeat::edgefeat_dim;
		}
	}

    return 0;
}
'''