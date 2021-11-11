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
from torch import optim
from gnn_libraries.pytorch_structure2vec.s2v_lib.embedding import EmbedMeanField, EmbedLoopyBP
from gnn_libraries.pytorch_structure2vec.s2v_lib.mlp import MLPRegression
from gnn_libraries.pytorch_structure2vec.harvard_cep.util import resampling_idxes
# other imports
import torch.nn as nn
import torch
import numpy as np
import random
import sys
import os


MODE = 'cpu'

class S2VRegressor(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, latent_dim=64, output_dim=1024, max_lv=3, hidden=100, gm='mean_field'):
        super(S2VRegressor, self).__init__()
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
        if MODE == 'gpu':
            node_feat = node_feat.cuda()
            edge_feat = edge_feat.cuda()
            labels = labels.cuda()
        embed = self.s2v(batch_graph, node_feat, edge_feat)
        
        return self.mlp(embed, labels)

# a function to train an instance of the s2v regressor above.
def TrainRegressor(regressor: S2VRegressor, data, 
                seed=0, saved_model='', save_dir='./s2v_model', phase='train', 
                batch_size=1, learning_rate=0.0001, num_epochs=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw_data_dict = load_raw_data()

    if MODE == 'gpu':
        regressor = regressor.cuda()
    if saved_model is not None and saved_model != '':
        if os.path.isfile(saved_model):
            print('loading model from %s' % saved_model)
            if MODE == 'cpu':
                regressor.load_state_dict(torch.load(saved_model, map_location=lambda storage, loc: storage))
            else:
                regressor.load_state_dict(torch.load(saved_model))

    if phase == 'test':
        test_data = MOLLIB.LoadMolGraph('test', raw_data_dict['test'])
        test_loss = loop_dataset(test_data, regressor, list(range(len(test_data))))
        print('\033[93maverage test loss: mae %.5f rmse %.5f\033[0m' % (test_loss[0], test_loss[1]))
        sys.exit()

    train_idxes = resampling_idxes(raw_data_dict)
    cooked_data_dict = {}
    for d in raw_data_dict:
        cooked_data_dict[d] = MOLLIB.LoadMolGraph(d, raw_data_dict[d])
    
    optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)
    iter_train = (len(train_idxes) + (batch_size - 1)) // batch_size

    best_valid_loss = None
    for epoch in range(num_epochs):        
        valid_interval = 10000
        for i in range(0, iter_train, valid_interval):
            avg_loss = loop_dataset(cooked_data_dict['train'], regressor, train_idxes, optimizer, start_iter=i, n_iters=valid_interval)
            print('\033[92maverage training of epoch %.2f: mae %.5f rmse %.5f\033[0m' % (epoch + min(float(i + valid_interval) / iter_train, 1.0), avg_loss[0], avg_loss[1]))

            valid_loss = loop_dataset(cooked_data_dict['valid'], regressor, list(range(len(cooked_data_dict['valid']))))
            print('\033[93maverage valid of epoch %.2f: mae %.5f rmse %.5f\033[0m' % (epoch + min(float(i + valid_interval) / iter_train, 1.0), valid_loss[0], valid_loss[1]))
            
            if best_valid_loss is None or valid_loss[0] < best_valid_loss:
                best_valid_loss = valid_loss[0]
                print('----saving to best model since this is the best valid loss so far.----')
                torch.save(regressor.state_dict(), save_dir + '/epoch-best.model')

        random.shuffle(train_idxes)
