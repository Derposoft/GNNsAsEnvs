# ai/ml imports
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import random
import json

# graph transformer imports
from model.nets.graph_transformer import GraphTransformerNet

# our code
from model.utils import get_cost_from_reward
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib

MAXIMUM_THEORETICAL_REWARD = 25

def initialize_train_artifacts(node_embedding_size, **kwargs):
    config_file = "configs/graph_transformer_config.json"
    with open(config_file) as f:
        config = json.load(f)
    net_params = config["net_params"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net_params["device"] = device
    net_params["gpu_id"] = config["gpu"]["id"]
    params = config["params"]
    net_params["batch_size"] = params["batch_size"]
    #params, net_params = main(return_config=True, config_file="model/graph_transformer_config.json")
    device = net_params["device"]
    net_params["node_embedding_size"] = node_embedding_size
    net_params["num_actions"] = 15 # TODO HARDCODED FOR NOW
    possible_kwargs = ["aggregation_fn", "L", "n_heads", "hidden_dim", "out_dim"]
    for p in possible_kwargs:
        if p in kwargs: net_params[p] = kwargs[p]

    # setting seeds
    if device.type == "cuda":
        torch.cuda.manual_seed(1234)

    model = GraphTransformerNet(net_params)
    #model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=params["init_lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params["lr_reduce_factor"],
        patience=params["lr_schedule_patience"],
        verbose=True,
    )
    
    return model, optimizer, scheduler


def optimize(optimizer, baseline, reward, ll, TEST_SETTINGS, num_steps=1, attention_input=None):
    local_max_theoretical_reward = MAXIMUM_THEORETICAL_REWARD
    if TEST_SETTINGS["normalize_losses_rewards_by_ep_length"]:
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
