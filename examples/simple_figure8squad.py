import gym
from random import randint
from sigma_graph.envs.figure8.maps.configs import ACTION_LOOKUP, ACTION_TURN_LOOKUP
import argparse

def print_lookup():
    print("'Move' actions: ", ACTION_LOOKUP)
    print("'Turn' actions: ", ACTION_TURN_LOOKUP)


def print_agents(env):
    print("Step #{}/{}".format(env.step_counter, env.max_step))
    print("\n### Team blue has {} agent(s)".format(env.num_blue))
    for _i in range(env.num_blue):
        _idx = env.team_blue[_i].get_index()
        _node, _dir = env.team_blue[_i].get_pos_dir()
        _look = ACTION_LOOKUP[_dir]
        _hp = env.team_blue[_i].get_health()
        _end = env.team_blue[_i].get_end_step()
        print("Agent blue #{} at index_in_route #{} @node: <{}> dir: <{}:\'{}\'> "
              "health: <{}> death_step: <{}>".format(_i, _idx, _node, _dir, _look, _hp, _end))
    print("\n### Team red has {} agent(s)".format(env.num_red))
    for _i in range(env.num_red):
        _node, _dir = env.team_red[_i].get_pos_dir()
        _look = ACTION_LOOKUP[_dir]
        _hp = env.team_red[_i].get_health()
        _dp = env.team_red[_i].damage_total()
        print("Agent red #{} @node: <{}> dir: <{}:\'{}\'> "
              "health: <{}> damage: <{}>".format(_i, _node, _dir, _look, _hp, _dp))
        print("mask: {}\nobs: {}\n".format(env.action_mask[_i], env.states[_i]))


def environment_example(config):
    n_episode = config.n_episode
    # init_red and init_blue should have number of agents dictionary elements if you want to specify it

    # [!!] remember to update this dict if adding new args in parser
    outer_configs = {"env_path": config.env_path, "n_red": config.n_red, "n_blue": config.n_blue,
                     "max_step": config.max_step, "init_health": config.init_health,
                     "obs_embed": config.obs_embed, "obs_dir": config.obs_dir, "obs_team": config.obs_team,
                     "obs_sight": config.obs_sight, "act_masked": config.act_masked,
                     "log_on": config.log_on, "log_path": config.log_path,
                     "init_red": config.init_red, "init_blue": config.init_blue}
    if hasattr(config, "penalty_stay"):
        outer_configs["penalty_stay"]: config.penalty_stay
    # "init_red": [{"pos": (11, 1), "dir": 1}, {"pos": None}, {"pos": "L", "dir": None}]
    ## init 'pos': tuple(x, z) or "L"/"R" region of the map

    ###===> Step 1. make
    env = gym.make('figure8squad-v0', **outer_configs)
    print("Env created with default env configs: {}\n".format(env.configs))
    print("loaded reward_configs:{} log_configs:{}\n".format(env.rewards, env.logs))
    print_lookup()
    print("\nNumbers of red:{} and blue:{}".format(env.num_red, env.num_blue))
    print("Observation shape: ", env.state_shape)

    # episode loop
    for ep in range(n_episode):
        print("\n###==> Eposide: {} of {}\n".format(ep + 1, n_episode))

        ###===> Step 2. intitial 'reset' before running 'step' functions
        # reset 'step_counter' and agent states for the next episode
        obs = env.reset()
        print("\nEnv after reset: new config:{} obs:{}".format(env.configs, obs))
        print_agents(env)

        # step loop
        for step in range(env.max_step):
            # generate random valid actions
            rand_actions = []
            for _i in range(len(env.learning_agent)):
                _valid = list(set([0 if _mask else _idx for _idx, _mask in enumerate(env.action_mask[_i][:5])]))
                _move = _valid[randint(0, len(_valid) - 1)]
                # print(env.action_mask[_i][:5], _valid, _move)
                _turn = randint(0, 2)
                rand_actions.append([_move, _turn])

            ###===> Step 3. run 'step' for 'max_step' times for an episode then 'reset'
            obs, rew, done, _ = env.step(rand_actions)
            # print updates
            print("#=> Step: {}/{} | Actions:{} | Rewards:{} | done: {}".format(env.step_counter, env.max_step,
                                                                                rand_actions, rew, done))
            # print_agents(env)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic configs
    parser.add_argument('--env_path', type=str, default='../', help='path of the project root')
    parser.add_argument('--n_red', type=int, default=2, help='numbers of red agent')
    parser.add_argument('--n_blue', type=int, default=1, help='numbers of blue agent')
    parser.add_argument('--n_episode', type=int, default=1, help='numbers of episode')
    parser.add_argument('--max_step', type=int, default=20, help='max step for each episode')
    parser.add_argument('--init_health', type=int, default=20, help='intitial HP for all agents')
    # advanced configs
    ''' feel free to add more paerser args [!!] keep in mind to update the 'outer_configs' if new args been added here
            All other valid config arguments including {
                _graph_args = {"env_map": 'S', "load_pkl": True}
                _config_args = ["damage_step", "damage_threshold_red", "damage_threshold_blue"]
                _reward_agent_step = ["reward_step_RB", "reward_step_BR", "reward_step_RR", "penalty_stay"]
                _reward_agent_done = ["reward_episode_lookup", "reward_faster_lookup"]
                _log_keys = ["log_"] + ["prefix", "on", "path", "save", "plot", "overview", "verbose"]
            }
    '''
    parser.add_argument('--obs_embed_on', dest="obs_embed", action='store_true', default=False, help='encoded embedding rather than raw one-hot POS')
    parser.add_argument('--obs_dir_off', dest="obs_dir", action='store_false', default=True, help='observation self 4 dir')
    parser.add_argument('--obs_team_off', dest="obs_team", action='store_false', default=True, help='observation teammates')
    parser.add_argument('--obs_sight_off', dest="obs_sight", action='store_false', default=True, help='observation in sight indicators')
    parser.add_argument('--act_masked_off', dest="act_masked", action='store_false', default=True, help='invalid action masking')
    parser.add_argument('--init_red', type=list, default=None, help='set init "pos" and "dir" for team red')
    parser.add_argument('--init_blue', type=list, default=None, help='set init "route" and "idx" for team blue')

    parser.add_argument('--log_on', dest="log_on", action='store_true', default=False, help='generate verbose logs')
    parser.add_argument('--log_path', type=str, default='logs/temp/', help='relative path to the project root')
    # parser.add_argument('--penalty_stay', type=int, default=-1, help='penalty for take stay action [0: "NOOP"]')
    config = parser.parse_args()
    print(config)

    # test run
    environment_example(config)


