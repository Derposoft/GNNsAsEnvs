import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""

from attention_study.model.nets.graph_transformer_edge_layer import GraphTransformerLayer
from attention_study.model.nets.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        #num_atom_type = net_params['num_atom_type'] # TODO if we want embedding layer
        #num_bond_type = net_params['num_bond_type']
        num_bond_type = net_params.get('num_bond_type', 1)
        node_embedding_size = net_params['node_embedding_size']
        num_actions = net_params.get('num_actions', 1)
        
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 37 # this is maximum graph size in the dataset

        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        # self.embedding_h = nn.Embedding(num_atom_type, hidden_dim) # TODO
        self.embedding_h = nn.Linear(node_embedding_size, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, num_actions)   # 1 out dim since regression problem
        #self.MLP_layer = MLPReadout(out_dim * 5, num_actions)   # version that uses 0/1/2/3/4 directions
        
    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None, agent_nodes=None, move_map=None):
        '''
        for more info on these inputs see: graph_transformer_rllib.py:forward()
        '''
        
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        return self.MLP_layer(hg)
        '''
        # TODO attempt 0; use curr node
        if agent_nodes != None:
            idxs = [g.batch_num_nodes()[0]*i+agent_nodes[i] for i in range(int(h.shape[0]/g.batch_num_nodes()[0].item()))]
            _embeddings = h[idxs, :]
            _outputs = self.MLP_layer(_embeddings)
            return _outputs
        '''
        '''
        # TODO attempt 1 code; use embeddings for directions from curr node
        batch_size = g.batch_num_nodes().shape[0]
        h = h.reshape([batch_size, -1, h.shape[-1]])
        # where we end up if we move in direction i
        m0 = agent_nodes
        #print(move_map)
        # return node in direction x if it exists; otherwise return the node itself
        def dir_node(dir, node): return move_map[node+1][dir]-1 if dir in move_map[node+1] else -1
        m1 = [dir_node(1, agent_nodes[i]) for i in range(len(agent_nodes))]
        m2 = [dir_node(2, agent_nodes[i]) for i in range(len(agent_nodes))]
        m3 = [dir_node(3, agent_nodes[i]) for i in range(len(agent_nodes))]
        m4 = [dir_node(4, agent_nodes[i]) for i in range(len(agent_nodes))]
        # embeddings of 0/1/2/3/4
        xs = range(batch_size)
        # collect embeddings
        o = h[xs,[m0,m1,m2,m3,m4],:]
        o = torch.permute(o, [1, 0, 2]).reshape([batch_size, -1])
        return self.MLP_layer(o)
        '''
        
        
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss