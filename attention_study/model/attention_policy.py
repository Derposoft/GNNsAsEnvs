'''
REQUIRES torch-geometric PACKAGE. INSTALLATION INSTRUCTIONS HERE:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}

simplified variant of https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py
with certain parts of network switched out for gnn layers. "policy.py" has the policy FCs switched
for gnns; "value.py" has the value FCs switched for gnns; "policy_value.py" has both branch's FCs
switched out for gats   .

most of this code is the same as the code on the linked github repo above; there was no reason to
rebuild one from scratch when one existed. 
'''
print('starting import process')
# RL/AI imports
#from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray.rllib.models.torch.torch_modelv2 as TMv2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer #AppendBiasLayer, \
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.agents import ppo
import torch.nn as nn
import torch
import gym
#from ray.tune.logger import pretty_print
print('imported rl/torch')

# our code imports
from attention_study.generate_baseline_metrics import parse_arguments, create_env_config
from sigma_graph.data.graph.skirmish_graph import MapInfo
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
from attention_study.model.utils import embed_obs_in_map, get_loc, load_edge_dictionary, \
    NETWORK_SETTINGS

# 3rd party library imports (s2v, attention model rdkit, etc?)
#from attention_study.model.s2v.s2v_graph import S2VGraph
#from attention_study.gnn_libraries.s2v.embedding import EmbedMeanField, EmbedLoopyBP
from attention_routing.nets.attention_model import AttentionModel
from attention_routing.problems.tsp.problem_tsp import TSP

print('imported our code+3rd party')

# other imports
import numpy as np
import os
import sys

class PolicyModel(TMv2.TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, map: MapInfo):#**kwargs):
        TMv2.TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
            model_config, name)
        nn.Module.__init__(self)

        # STEP 0: set config
        # original config
        hiddens = list(model_config.get("fcnet_hiddens", [])) + \
            list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers") # this is usually 0
        self.free_log_std = model_config.get("free_log_std") # skip worrying about log std
        
        # my settings
        gnns = 10
        self.has_final_layer = NETWORK_SETTINGS['has_final_layer']
        self.map = map # map+edges parsing
        self.acs_edges_dict = load_edge_dictionary(self.map.g_acs.adj)
        self.vis_edges_dict = load_edge_dictionary(self.map.g_vis.adj)
        self.edge_to_action = None # TODO INITIALIZA THIS

        # STEP 1: build policy net -- GAT + FF
        with open('./args.json') as file:
            import json
            file = dict(json.load(file))
            self.attention = AttentionModel(
                embedding_dim=file['embedding_dim'],
                hidden_dim=file['hidden_dim'],
                problem=TSP
            )
            self.attention.set_decode_type('greedy')
        print('attention model initiated')
        # create logits if we are using logits
        #prev_layer_size = int(np.product(obs_space.shape))
        #prev_layer_size = int(map.get_graph_size())
        self._logits = None
        if self.has_final_layer:
            self._logits = SlimFC(
                in_size=map.get_graph_size(),
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None)                

        # STEP 2: build value net
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
        prev_layer_size = hiddens[-1]
        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        # Holds the current "base" output (before logits layer)
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None
        print('policy model initiated')

    @override(TMv2.TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType):
        obs = input_dict['obs_flat'].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        
        # run attention model
        attention_input = embed_obs_in_map(obs, self.map)
        agent_nodes = [get_loc(gx, self.map.get_graph_size()) for gx in obs]
        _, _, log_ps = self.attention(attention_input, edges=self.acs_edges_dict, agent_nodes=agent_nodes, return_log_p=True)

        # decode actions
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = log_ps
        logits = None
        if self._logits:
            # action decoding via logits
            logits = self._logits(self._features)
        else:
            actions = []
            # move_action decoding. get max prob moves from map
            transformed_features = self._features.clone()
            transformed_features[transformed_features == 0] = -float('inf')
            optimal_destination = torch.argmax(transformed_features, dim=1)
            # collect move actions
            for i in range(len(agent_nodes)):
                curr_loc = agent_nodes[i] + 1
                next_loc = optimal_destination[i].item() + 1
                move_action = self.map.g_acs.adj[curr_loc][next_loc]['action']
                look_action = 1 # TODO!!!!!!!!
                action = Figure8SquadRLLib.convert_multidiscrete_action_to_discrete(move_action, look_action)
                actions.append(action)
            logits = torch.tensor(np.eye(self.num_outputs)[actions])
        
        return logits, state

    @override(TMv2.TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)


if __name__ == "__main__":
    # register our model (put in an __init__ file later)
    # https://docs.ray.io/en/latest/rllib-models.html#customizing-preprocessors-and-models
    ModelCatalog.register_custom_model("policy_model", PolicyModel)

    # STEP 0: parse arguments
    print('creating config')
    parser = parse_arguments()
    config = parser.parse_args()
    outer_configs, n_episodes = create_env_config(config)
    
    # STEP 1: test custom model above with an rllib trainer
    print('creating trainer')
    # (using ppo to test)
    def create_ppo_config(outer_configs):
        # policy mapping function
        # from https://medium.com/@vermashresth/craft-and-solve-multi-agent-problems-using-rllib-and-tensorforce-a3bd1bb6f556
        #setup_env = gym.make('figure8squad-v3', **outer_configs)
        setup_env = Figure8SquadRLLib(outer_configs)
        obs_space = setup_env.observation_space
        act_space = setup_env.action_space
        policies = {}
        for agent_id in setup_env.learning_agent:
            policies[str(agent_id)] = (None, obs_space, act_space, {})
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return str(agent_id)
        # create trainer config
        ppo_extra_config_settings = {
            "env": Figure8SquadRLLib,
            "env_config": {
                **outer_configs
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "policy_model",
                # Extra kwargs to be passed to your model's c'tor.
                "custom_model_config": {
                    "map": setup_env.map
                },
            },
            "num_workers": 1,  # parallelism
            "framework": "torch",
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "evaluation_interval": 1,
            "evaluation_num_episodes": 10,
            "evaluation_num_workers": 1,
            "rollout_fragment_length": 200,
            "train_batch_size": 200
        }
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(ppo_extra_config_settings)
        ppo_config["lr"] = 1e-3 # fixed lr instead of schedule, tune this
        return ppo_config
    ppo_trainer = ppo.PPOTrainer(config=create_ppo_config(outer_configs), env=Figure8SquadRLLib)

    print('trainer created')
    ppo_trainer.train()
    

'''
# GARBAGE ZONE

# 2. 
# 1: use s2v to create node embeddings (inspiration from Khalil et al, 2017)
#input_graph = S2VGraph(len(self.nodes), len(self.edge_pairs), self.edge_pairs)
#embedding = self.s2v([input_graph], self.nodes, self.edge_pairs)
#print(embedding)
# 
'''