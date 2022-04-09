'''
base class from https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py
'''
# RL/AI imports
import ray.rllib.models.torch.torch_modelv2 as TMv2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
import torch.nn as nn
import torch
import gym
import dgl

# our code imports
from sigma_graph.data.graph.skirmish_graph import MapInfo
from sigma_graph.envs.figure8 import default_setup as env_setup
from model.utils import GRAPH_OBS_TOKEN, count_model_params, efficient_embed_obs_in_map, get_loc
from model.graph_transformer_model import initialize_train_artifacts as initialize_graph_transformer

# other imports
import numpy as np

class HybridPolicy(TMv2.TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, map: MapInfo, **kwargs):
        TMv2.TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
            model_config, name)
        nn.Module.__init__(self)
        # config
        hiddens = list(model_config.get("fcnet_hiddens", [])) + \
            list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers") # this is usually 0
        self.free_log_std = model_config.get("free_log_std") # skip worrying about log std
        #self.use_mean_embed = kwargs['use_mean_embed'] # obs information
        self.map = map
        self.num_red = kwargs['nred']
        self.num_blue = kwargs['nblue']
        self_shape, red_shape, blue_shape = env_setup.get_state_shapes(self.map.get_graph_size(), self.num_red, self.num_blue, env_setup.OBS_TOKEN)
        self.obs_shapes = [self_shape, red_shape, blue_shape, self.num_red, self.num_blue]
        #self.map.g_acs.add_node(0) # dummy node that we'll use later
        '''
        self.hidden_proj_sizes = [50, 30]
        self.GAT_LAYERS = 6
        self.N_HEADS = 4
        self.HIDDEN_DIM = 4
        '''
        self.hidden_proj_sizes = [45, 45]
        self.GAT_LAYERS = 4
        self.N_HEADS = 4
        self.HIDDEN_DIM = 4

        # map info
        self.move_map = {} # movement dictionary: d[node][direction] = newnode. newnode is -1 if direction is not possible from node
        for n in self.map.g_acs.adj:
            self.move_map[n] = {}
            ms = self.map.g_acs.adj[n]
            for m in ms:
                dir = ms[m]['action']
                self.move_map[n][dir] = m
        
        # actor (attention model)
        self.gats, _, _ = initialize_graph_transformer(GRAPH_OBS_TOKEN['embedding_size'], readout='none', L=self.GAT_LAYERS, n_heads=self.N_HEADS, hidden_dim=self.HIDDEN_DIM, out_dim=self.HIDDEN_DIM)
        self.o_proj = nn.Sequential(
            nn.Linear(self.gats.embedding_h.out_features+self_shape+red_shape+blue_shape, self.hidden_proj_sizes[0]),
            nn.Tanh(),
            nn.Linear(self.hidden_proj_sizes[0], self.hidden_proj_sizes[1]),
            nn.Tanh(),
            nn.Linear(self.hidden_proj_sizes[1], self.num_outputs)
        )

        # critic
        self._value_branch_separate = None
        # create value network with equal number of hidden layers as policy net
        if not self.vf_share_layers:
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0)))
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)
        # layer which outputs 1 value
        #prev_layer_size = hiddens[-1] if self._value_branch_separate else self.map.get_graph_size()
        prev_layer_size = hiddens[-1] if self._value_branch_separate else int(action_space.n)
        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        # Holds the current "base" output (before logits layer)
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

        count_model_params(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.cache = {} # minor speedup (~15%) of training

    @override(TMv2.TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType):
        #start_time = time.time()
        obs = input_dict['obs_flat'].float()
        # transform obs to graph
        attention_input = efficient_embed_obs_in_map(obs, self.map, self.obs_shapes)
        agent_nodes = [get_loc(gx, self.map.get_graph_size()) for gx in obs]
    
        if len(obs) not in self.cache:
            batch_graphs = []
            for i in range(len(obs)):
                batch_graphs.append(dgl.from_networkx(self.map.g_acs))
            batch_graphs = dgl.batch(batch_graphs)
            batch_graphs = batch_graphs.to(self.device)
            self.cache[len(obs)] = batch_graphs.clone()
        else:
            batch_graphs = self.cache[len(obs)].clone()
        batch_graphs.ndata['feat'] = attention_input.reshape([-1, GRAPH_OBS_TOKEN['embedding_size']])
        
        # inference
        batch_x = batch_graphs.ndata['feat']
        batch_e, batch_lap_enc, batch_wl_pos_enc = None, None, None
        gat_output = self.gats(batch_graphs, batch_x, batch_e, batch_lap_enc, batch_wl_pos_enc, agent_nodes, self.move_map)
        logits = self.o_proj(torch.cat([gat_output, obs], dim=1))

        # return
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = logits
        #print('forward takes', time.time()-start_time)
        return logits, state

    @override(TMv2.TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)
