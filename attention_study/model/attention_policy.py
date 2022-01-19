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
# RL/AI imports
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray.rllib.models.torch.torch_modelv2 as TMv2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.agents import ppo
import torch.nn as nn
import torch
import gym
from ray.tune.logger import pretty_print

# our code imports
from attention_study.generate_baseline_metrics import parse_arguments, create_env_config
from sigma_graph.data.graph.skirmish_graph import MapInfo
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
from attention_study.model.utils import embed_obs_in_map

# 3rd party library imports (s2v, attention model rdkit, etc?)
#from attention_study.model.s2v.s2v_graph import S2VGraph
#from attention_study.gnn_libraries.s2v.embedding import EmbedMeanField, EmbedLoopyBP
from attention_routing.nets.attention_model import AttentionModel
from attention_routing.problems.tsp.problem_tsp import TSP

# other imports
import numpy as np
import os
import time
import sys

NUM_NODE_FEATURES = 1 # DETERMINED BY S2V!!!!!

class PolicyModel(TMv2.TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, map: MapInfo):#**kwargs):
        TMv2.TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
            model_config, name)
        nn.Module.__init__(self)

        # STEP 0: parse model_config args
        # STEP 0.1: parse boilerplate
        gnns = 10
        hiddens = list(model_config.get("fcnet_hiddens", [])) + \
            list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers") # this is usually 0
        self.free_log_std = model_config.get("free_log_std") # skip worrying about log std
        self.map = map

        # STEP 1: build policy net -- GAT + FF
        with open('./args.json') as file:
            import json
            file = dict(json.load(file))
            self.attention = AttentionModel(
                embedding_dim=file['embedding_dim'],
                hidden_dim=file['hidden_dim'],
                problem=TSP)

        # STEP 2: fc layers post attention layers
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # create layers 0->n-1
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size
        # create last layer
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation))
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None)
            else:
                self.num_outputs = (
                    [int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        self._hidden_layers = nn.Sequential(*layers)

        # STEP 3: build value net
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

    @override(TMv2.TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType):
        # 1. run attention model's forward
        # input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        obs = input_dict["obs_flat"].float()
        attention_input = embed_obs_in_map(obs, self.map)
        sys.exit()
        cost, log_likelihood = self.attention.forward(attention_input)

        # 2: run thru fc layers
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