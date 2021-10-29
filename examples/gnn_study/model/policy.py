# RL/AI imports
import ray.rllib.models.torch.torch_modelv2 as TMv2
import ray.rllib.models.tf.tf_modelv2 as TFv2
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.rllib.agents import ppo
import gym
import torch.nn as nn
# our code imports
from examples.gnn_study.generate_baseline_metrics import parse_arguments, create_env_config
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
# other imports
import os

class Policy(TMv2.TorchModelV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = nn.Linear(len(self.obs_space), 128)
        self.h1 = nn.Linear(128, 64)
        self.h2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, len(self.action_space))
    # forward function
    def forward(self, obs):
        x = self.input(obs)
        x = self.h1(nn.ReLU(x))
        x = self.h2(nn.ReLU(x))
        x = self.output(nn.ReLU(x))
        return x


if __name__ == "__main__":
    # STEP 0: parse arguments
    parser = parse_arguments()
    config = parser.parse_args()
    outer_configs, n_episodes = create_env_config(config)
    #policy = Policy()
    
    # STEP 1: test custom model above with an rllib trainer
    # (using ppo to test)
    def create_ppo_config(outer_configs):
        # policy mapping function
        # from https://medium.com/@vermashresth/craft-and-solve-multi-agent-problems-using-rllib-and-tensorforce-a3bd1bb6f556
        setup_env = gym.make('figure8squad-v3', **outer_configs)
        obs_space = setup_env.observation_space
        act_space = setup_env.action_space
        policies = {}
        for agent_id in setup_env.learning_agent:
            policies[str(agent_id)] = (None, obs_space, act_space, {})
        #policies['default_policy'] = (None, obs_space, act_space, {}) # necessary for impala
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
            "model": Policy,
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
    print('ppo trainer loaded...')
