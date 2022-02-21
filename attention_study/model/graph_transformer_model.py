import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

from graph_transformer.nets.molecules_graph_regression.load_net import gnn_model 

# our code
from attention_study.model.utils import get_cost_from_reward

MAXIMUM_THEORETICAL_REWARD = 25

def initialize_train_artifacts(MODEL_NAME, dataset, params, net_params, dirs):
    DATASET_NAME = dataset.name
    
    if net_params['lap_pos_enc']:
        st = time.time()
        print("[!] Adding Laplacian positional encoding.")
        dataset._add_laplacian_positional_encodings(net_params['pos_enc_dim'])
        print('Time LapPE:',time.time()-st)
        
    if net_params['wl_pos_enc']:
        st = time.time()
        print("[!] Adding WL positional encoding.")
        dataset._add_wl_positional_encodings()
        print('Time WL PE:',time.time()-st)
    
    if net_params['full_graph']:
        st = time.time()
        print("[!] Converting the given graphs to full graphs..")
        dataset._make_full_graph()
        print('Time taken to convert to full graphs:',time.time()-st)   
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    
    return model, optimizer, None, scheduler, writer


def optimize(optimizer, baseline, reward, ll, TEST_SETTINGS, num_steps=1, attention_input=None):
    local_max_theoretical_reward = MAXIMUM_THEORETICAL_REWARD
    if TEST_SETTINGS['normalize_losses_rewards_by_ep_length']:
        reward /= num_steps
        ll /= num_steps
        local_max_theoretical_reward /= num_steps
    # set costs
    model_cost = get_cost_from_reward(reward)
    #bl_val = get_cost_from_reward(local_max_theoretical_reward)
    #bl_loss = 0
    #reinforce_loss = ((bl_val - model_cost) * ll).mean()
    #loss = reinforce_loss + bl_loss
    net_loss = nn.L1Loss()()
    optimizer.zero_grad()
    loss = model_cost
    loss.backward()
    optimizer.step()
    return None, loss
