import gym
from random import randint
from sigma_graph.envs.figure8.action_lookup import MOVE_LOOKUP, TURN_90_LOOKUP
import argparse


def print_lookup():
    print("'Move' actions: ", MOVE_LOOKUP)
    print("'Turn' actions: ", TURN_90_LOOKUP)


def print_agents(env):
    # print("Step #{}/{}".format(env.step_counter, env.max_step))
    print("## Team red has {} agent(s)".format(env.num_red))
    for _i in range(env.num_red):
        _id = env.team_red[_i].get_id()
        _node, _dir = env.team_red[_i].get_pos_dir()
        _look = MOVE_LOOKUP[_dir]
        _hp = env.team_red[_i].get_health()
        _dp = env.team_red[_i].damage_total()
        print("# Agent red #{} @node: <{}> dir: <{}:\'{}\'> "
              "health: <{}> damage: <{}>".format(_id, _node, _dir, _look, _hp, _dp))
        print("mask: {}\nobs: {}".format(env.action_mask[_i], env.states[_i]))
    print("## Team blue has {} agent(s)".format(env.num_blue))
    for _i in range(env.num_blue):
        _id = env.team_blue[_i].get_id()
        _idx = env.team_blue[_i].get_index()
        _node, _dir = env.team_blue[_i].get_pos_dir()
        _look = MOVE_LOOKUP[_dir]
        _hp = env.team_blue[_i].get_health()
        _end = env.team_blue[_i].get_end_step()
        print("# Agent blue #{} at pos_index #{} @node: <{}> dir: <{}:\'{}\'> "
              "health: <{}> death_step: <{}>\n".format(_id, _idx, _node, _dir, _look, _hp, _end))


def environment_example(config):
    n_episode = config.n_episode
    # init_red and init_blue should have number of agents dictionary elements if you want to specify it

    # [!!] remember to update this dict if adding new args in parser
    outer_configs = {"env_path": config.env_path, "max_step": config.max_step, "act_masked": config.act_masked,
                     "n_red": config.n_red, "n_blue": config.n_blue,
                     "init_red": config.init_red, "init_blue": config.init_blue,
                     "init_health_red": config.init_health, "init_health_blue": config.init_health,
                     "obs_embed": config.obs_embed, "obs_dir": config.obs_dir, "obs_team": config.obs_team,
                     "obs_sight": config.obs_sight,
                     "log_on": config.log_on, "log_path": config.log_path,
                     # "reward_step_on": False, "reward_episode_on": True, "episode_decay_soft": True,
                     # "health_lookup": {"type": "table", "reward": [8, 4, 2, 0], "damage": [0, 1, 2, 100]},
                     # "faster_lookup": {"type": "none"},
                     }
    ## i.e. init_red 'pos': tuple(x, z) or "L"/"R" region of the map
    # "init_red": [{"pos": (11, 1), "dir": 1}, {"pos": None}, {"pos": "L", "dir": None}]
    if hasattr(config, "penalty_stay"):
        outer_configs["penalty_stay"] = config.penalty_stay
    if hasattr(config, "threshold_blue"):
        outer_configs["threshold_damage_2_blue"] = config.threshold_blue
    if hasattr(config, "threshold_red"):
        outer_configs["threshold_damage_2_red"] = config.threshold_red

    ###===> Step 1. make env
    env = gym.make('figure8squad-v3', **outer_configs)
    # print inits
    print(f"Env created with default env configs: {env.configs}\n")
    print(f"loaded reward_configs:{env.rewards} log_configs:{env.logs}\n")
    print_lookup()
    print(f"\nNumbers of red:{env.num_red} and blue:{env.num_blue}")
    print("Observation shape: ", env.state_shape)

    # episode loop
    for ep in range(n_episode):
        print(f"\n#####===> Episode: {ep + 1} of {n_episode}\n")

        ###===> Step 2. intitial 'reset' before running 'step' functions
        # reset 'step_counter' and agent states for the next episode
        obs = env.reset()
        # print configs
        print(f"Env after reset: new config:{env.configs} obs:{obs}")
        print_agents(env)

        # step loop
        for step in range(env.max_step):
            rand_actions = []
            # generate random valid actions
            for _i in range(len(env.learning_agent)):
                _valid = list(set([0 if _mask else _idx for _idx, _mask in enumerate(env.action_mask[_i][:5])]))
                _move = _valid[randint(0, len(_valid) - 1)]
                _turn = randint(0, 2)
                rand_actions.append([_move, _turn])

            ###===> Step 3. run 'step' for 'max_step' times for an episode then 'reset'
            obs, rew, done, _ = env.step(rand_actions)
            # print updates
            print("###==> Step: {}/{} | Actions:{} | Rewards:{} | done: {}".format(env.step_counter, env.max_step,
                                                                                   rand_actions, rew, done))
            # end this episode if we are all done
            if all(done):
                break
        print_agents(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic configs
    parser.add_argument('--env_path', type=str, default='../', help='path of the project root')
    parser.add_argument('--n_red', type=int, default=2, help='numbers of red agent')
    parser.add_argument('--n_blue', type=int, default=1, help='numbers of blue agent')
    parser.add_argument('--n_episode', type=int, default=1, help='numbers of episode')
    parser.add_argument('--max_step', type=int, default=20, help='max step for each episode')
    parser.add_argument('--init_health', type=int, default=20, help='initial HP for all agents')
    # advanced configs
    parser.add_argument('--obs_embed_on', dest="obs_embed", action='store_true', default=False,
                        help='encoded embedding rather than raw one-hot POS')
    parser.add_argument('--obs_dir_off', dest="obs_dir", action='store_false', default=True,
                        help='observation self 4 dir')
    parser.add_argument('--obs_team_off', dest="obs_team", action='store_false', default=True,
                        help='observation teammates')
    parser.add_argument('--obs_sight_off', dest="obs_sight", action='store_false', default=True,
                        help='observation in sight indicators')
    parser.add_argument('--act_masked_off', dest="act_masked", action='store_false', default=True,
                        help='invalid action masking')
    parser.add_argument('--init_red', type=list, default=None, help='set init "pos" and "dir" for team red')
    parser.add_argument('--init_blue', type=list, default=None, help='set init "route" and "idx" for team blue')
    parser.add_argument('--log_on', dest="log_on", action='store_true', default=False, help='generate verbose logs')
    parser.add_argument('--log_path', type=str, default='logs/temp/', help='relative path to the project root')
    ''' feel free to add more parser args [!!] keep in mind to update the 'outer_configs' if new args been added here
        All other valid config arguments including {
            _graph_args = {"map_id": 'S', "load_pickle": True}
            _config_args = ["damage_maximum", "damage_threshold_red", "damage_threshold_blue"]
            INTERACT_LOOKUP = {
                "sight_range": -1,  # -1 for unlimited range
                "engage_range": 25,
                "engage_behavior": {"damage": 1, "probability": 1.0},
            }
            INIT_LOGS = {
                "log_on": False, "log_path": "logs/", "log_prefix": "log_", "log_overview": "reward_episodes.txt",
                "log_verbose": False, "log_plot": False, "log_save": True,
            }
        }
    '''
    # additional (comment out if not needed)
    parser.add_argument('--penalty_stay', type=int, default=0, help='penalty for take stay action [0: "NOOP"]')
    parser.add_argument('--threshold_blue', default=2)
    parser.add_argument('--threshold_red', default=5)

    # demo run
    config = parser.parse_args()
    environment_example(config)
