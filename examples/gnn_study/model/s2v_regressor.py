'''
A (mostly similar) variant of the s2v regressor provided by Dai et. al at the following link:
https://github.com/Hanjun-Dai/pytorch_structure2vec


CREDIT:
@article{dai2016discriminative,
  title={Discriminative Embeddings of Latent Variable Models for Structured Data},
  author={Dai, Hanjun and Dai, Bo and Song, Le},
  journal={arXiv preprint arXiv:1603.05629},
  year={2016}
}
'''
# our imports

# Dai et. al imports
from gnn_libraries.pytorch_structure2vec.s2v_lib.embedding import EmbedMeanField, EmbedLoopyBP
from gnn_libraries.pytorch_structure2vec.s2v_lib.mlp import MLPRegression
# other imports
import torch.nn as nn
import sys

class S2VRegressor(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, latent_dim=64, output_dim=1024, max_lv=3, hidden=100, gm='mean_field'):
        super(S2VRegressor, self).__init__()
        self.mode = 'cpu'
        if gm == 'mean_field':
            model = EmbedMeanField
        elif gm == 'loopy_bp':
            model = EmbedLoopyBP
        else:
            print('unknown gm %s' % gm)
            sys.exit()

        self.s2v = model(latent_dim=latent_dim, 
                        output_dim=output_dim,
                        num_node_feats=num_node_feats, 
                        num_edge_feats=num_edge_feats,
                        max_lv=max_lv)
        self.mlp = MLPRegression(input_size=output_dim, hidden_size=hidden)

    def forward(self, batch_graph): 
        node_feat, edge_feat, labels = batch_graph # MOLLIB.PrepareFeatureLabel(batch_graph) TODO figure out what this function does and emulate it for our data
        if self.mode == 'gpu':
            node_feat = node_feat.cuda()
            edge_feat = edge_feat.cuda()
            labels = labels.cuda()
        embed = self.s2v(batch_graph, node_feat, edge_feat)
        
        return self.mlp(embed, labels)