"""
base class from https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py
policy: this is a "hybrid model" which has one part FC and one part graph transformer for a policy
value: standard FC value

# params: 95328
"""
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
from model.graph_transformer_model import initialize_train_artifacts as initialize_graph_transformer
import model.utils as utils

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
        #self.use_mean_embed = kwargs["use_mean_embed"] # obs information
        self.map = map
        self.num_red = kwargs["nred"]
        self.num_blue = kwargs["nblue"]
        self.aggregation_fn = kwargs["aggregation_fn"]
        self.hidden_size = kwargs["hidden_size"]
        self_shape, blue_shape, red_shape = env_setup.get_state_shapes(
            self.map.get_graph_size(), self.num_red, self.num_blue, env_setup.OBS_TOKEN
        )
        self.obs_shapes = [self_shape, blue_shape, red_shape, self.num_red, self.num_blue]
        
        """
        self.hidden_proj_sizes = [45, 45]
        self.GAT_LAYERS = 4
        self.N_HEADS = 4
        self.HIDDEN_DIM = 4
        """
        #self.hidden_proj_sizes = [150, 150]
        self.hiddens = [self.hidden_size, self.hidden_size]
        #self.hidden_proj_sizes = [177, 177]
        self.GAT_LAYERS = 4
        self.N_HEADS = 4
        self.HIDDEN_DIM = 4

        # map info
        self.move_map = utils.create_move_map(map.g_acs)
        
        # policy -- gats and mlp
        self.gats, _, _ = initialize_graph_transformer(
            utils.NODE_EMBED_SIZE,
            aggregation_fn=self.aggregation_fn,
            L=self.GAT_LAYERS,
            n_heads=self.N_HEADS,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.HIDDEN_DIM,
        )
        self._hiddens, self._logits = utils.create_policy_fc(
            hiddens=self.hiddens,
            activation=activation,
            num_outputs=num_outputs,
            no_final_linear=no_final_linear,
            num_inputs=int(np.product(obs_space.shape))+self.gats.num_actions,
        )
        # value
        self._value_branch, self._value_branch_separate = utils.create_value_branch(
            num_inputs=int(np.product(obs_space.shape)),
            num_outputs=num_outputs,
            vf_share_layers=self.vf_share_layers,
            activation=activation,
            hiddens=utils.VALUE_HIDDENS,
        )
        # hold previous inputs
        self._features = None
        self._last_flat_in = None

        utils.count_model_params(self, print_model=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.cache = {} # minor speedup (~15%) of training

    @override(TMv2.TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType):
        #start_time = time.time()
        obs = input_dict["obs_flat"].float()
        # transform obs to graph
        attention_input = utils.efficient_embed_obs_in_map(obs, self.map, self.obs_shapes)
        agent_nodes = [utils.get_loc(gx, self.map.get_graph_size()) for gx in obs]
    
        if len(obs) not in self.cache:
            batch_graphs = []
            for i in range(len(obs)):
                batch_graphs.append(dgl.from_networkx(self.map.g_acs))
            batch_graphs = dgl.batch(batch_graphs)
            batch_graphs = batch_graphs.to(self.device)
            self.cache[len(obs)] = batch_graphs.clone()
        else:
            batch_graphs = self.cache[len(obs)].clone()
        batch_graphs.ndata["feat"] = attention_input.reshape(
            [-1, utils.NODE_EMBED_SIZE]
        )
        
        # inference
        batch_x = batch_graphs.ndata["feat"]
        batch_e, batch_lap_enc, batch_wl_pos_enc = None, None, None
        gat_output = self.gats(
            batch_graphs,
            batch_x,
            batch_e,
            batch_lap_enc,
            batch_wl_pos_enc,
            agent_nodes,
            self.move_map
        )
        self._features = self._hiddens(torch.cat([gat_output, obs], dim=1))

        # return
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        logits = self._logits(self._features)
        #print("forward takes", time.time()-start_time)
        return logits, state

    @override(TMv2.TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        if not self._value_branch:
            return torch.Tensor([0]*len(self._features))
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)
