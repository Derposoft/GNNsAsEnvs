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
from attention_study.model.utils import embed_obs_in_map, get_loc, NETWORK_SETTINGS

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

        # STEP 0: parse model_config args
        # STEP 0.1: parse args
        gnns = 10
        hiddens = list(model_config.get("fcnet_hiddens", [])) + \
            list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers") # this is usually 0
        self.free_log_std = model_config.get("free_log_std") # skip worrying about log std
        self.map = map # map+edges parsing
        self.acs_edges = []
        self.vis_edges = []
        for k, v in zip(self.map.g_acs.adj.keys(), self.map.g_acs.adj.values()):
            self.acs_edges += [[k, vi] for vi in v.keys()]
        for k, v in zip(self.map.g_acs.adj.keys(), self.map.g_acs.adj.values()):
            self.vis_edges += [[k, vi] for vi in v.keys()]
        self.acs_edges_dict = {}
        self.vis_edges_dict = {}
        for edge in self.acs_edges:
            if edge[0] not in self.acs_edges_dict: self.acs_edges_dict[edge[0]] = set([])
            self.acs_edges_dict[edge[0]].add(edge[1])
        for edge in self.vis_edges:
            if edge[0] not in self.vis_edges_dict: self.vis_edges_dict[edge[0]] = set([])
            self.vis_edges_dict[edge[0]].add(edge[1])

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

        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None
        if NETWORK_SETTINGS['has_final_layer']:
            self._logits = SlimFC(
                in_size=prev_layer_size,
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
        
        # run attention model
        attention_input = embed_obs_in_map(obs, self.map)
        agent_nodes = [get_loc(gx, self.map.get_graph_size()) for gx in obs]
        costs, lls, log_ps = self.attention(attention_input, edges=self.acs_edges_dict, agent_nodes=agent_nodes, return_log_p=True)
        # TODO get this to output log_p instead; use model._inner?
        print(log_ps)
        sys.exit()
        # add mask
        
        # run thru fc layers
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)#, self.nodes, self.edge_index)
        logits = self._logits(self._features) if self._logits else \
            self._features
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