"""
base class from https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py
"""
import torch.nn as nn
import torch
import ray.rllib.models.torch.torch_modelv2 as TMv2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
import gym
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

import dgl
from torch_geometric.nn.conv import GATConv
import networkx as nx

from sigma_graph.data.graph.skirmish_graph import MapInfo
from sigma_graph.envs.figure8 import default_setup as env_setup
from sigma_graph.envs.figure8.action_lookup import MOVE_LOOKUP
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
import model.utils as utils


class GATPolicy(TMv2.TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        map: MapInfo,
        **kwargs
    ):
        TMv2.TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        nn.Module.__init__(self)

        """
        values that we need to instantiate and use GNNs
        """
        (
            hiddens,
            activation,
            no_final_linear,
            self.vf_share_layers,
            self.free_log_std,
        ) = utils.parse_config(model_config)
        self.map = map
        self.num_red = kwargs["nred"]
        self.num_blue = kwargs["nblue"]
        self_shape, red_shape, blue_shape = env_setup.get_state_shapes(
            self.map.get_graph_size(),
            self.num_red,
            self.num_blue,
            env_setup.OBS_TOKEN,
        )
        self.obs_shapes = [
            self_shape,
            red_shape,
            blue_shape,
            self.num_red,
            self.num_blue,
        ]
        self.adjacency = []
        for n in map.g_acs.adj:
            ms = map.g_acs.adj[n]
            for m in ms:
                self.adjacency.append([n-1, m-1])
        self.adjacency = torch.LongTensor(self.adjacency).t().contiguous()
        self._features = None  # current "base" output before logits
        self._last_flat_in = None  # last input

        """
        instantiate policy and value networks
        """
        self.GAT_LAYERS = 8
        self.N_HEADS = 8
        self.HIDDEN_DIM = 8
        self.gats = [
            GATConv(
                in_channels=utils.NODE_EMBED_SIZE if i == 0 else self.HIDDEN_DIM*self.N_HEADS,
                out_channels=self.HIDDEN_DIM,
                heads=self.N_HEADS,
            )
            for i in range(self.GAT_LAYERS)
        ]
        self._value_branch, self._value_branch_separate = utils.create_value_branch(
            obs_space=obs_space,
            action_space=action_space,
            vf_share_layers=self.vf_share_layers,
            activation=activation,
            hiddens=utils.VALUE_HIDDENS,
        )

        """
        produce debug output and ensure that model is on right device
        """
        utils.count_model_params(self)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    @override(TMv2.TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ):
        # transform obs to graph (for pyG, also do list[data]->torch_geometric.Batch)
        obs = input_dict["obs_flat"].float()
        x = utils.efficient_embed_obs_in_map(obs, self.map, self.obs_shapes)
        #x = list(x)
        #x = [Data(x_, self.adjacency) for x_ in x] 
        #x = Batch.from_data_list(x)
        #print(x)
        
        # inference
        for conv in self.gats:
            print("STARTED LAYER", x.shape)
            x = torch.stack([conv(_x, self.adjacency) for _x in x], dim=0)
            #x = conv(x, self.adjacency)
            print("DONE LAYER", x.shape)
        logits = x

        # return
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = logits
        return logits, state

    @override(TMv2.TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)
