import numpy as np
from random import randint

import gym
from gym import spaces

from ..utils.multiagent_space import ActionSpaces, ObservationSpaces
from ..utils.multiagent_agent_base import MAgent
from ...data.file_manager import load_graph_files, save_log_2_file, log_done_reward

from .maps.skirmish_graph import MapInfo, RouteInfo
from .maps.configs import ACTION_LOOKUP, ACTION_TURN_LOOKUP, TURN_L, TURN_R, INTERACT_LOOKUP, INIT_POS_LOOKUP
from .maps.data_helper import get_node_name_from_pos_abs, check_pos_abs_range, get_pos_norms, get_emb_from_name
from .rewards.rewards_simple import get_step_engage, get_step_overlay, get_episode_reward_agent, get_episode_reward_team


class Figure8Squad(gym.Env):
    def __init__(self, max_step=40, n_red=2, n_blue=1, init_health=100,
                 env_path='./', env_map='S', load_pkl=True, act_masked=True,
                 obs_embed=False, obs_dir=True, obs_team=True, obs_sight=True, **args):
        # setup env configs
        self.max_step = max_step
        self.step_counter = 0
        self.done_counter = 0
        self.configs = {"init_health": init_health, "env_path": env_path, "map_id": env_map, "load_pickle": load_pkl,
                        "obs_embed": obs_embed, "obs_dir": obs_dir, "obs_team": obs_team, "obs_sight": obs_sight}
        self.rewards = {}
        self.logger = False
        self.logs = {}
        # set default local configs and parse outer arguments
        self.num_red = n_red
        self.num_blue = n_blue
        self._init_env_config(**args)

        # env map and patrol routes (loading connectivity & visibility graphs)
        self.map = MapInfo()
        # list of RouteInfo instances
        self.routes = []
        self._load_map_data()

        # lists of agent instances
        self.team_red = []
        self.team_blue = []
        # default training agents are agents in team red
        self.learning_agent = []
        self._init_agents()

        # init spaces
        self.action_space = ActionSpaces([spaces.MultiDiscrete([len(ACTION_LOOKUP), len(ACTION_TURN_LOOKUP)])
                                          for _ in range(len(self.learning_agent))])
        # default: all agents have identical observation shape
        self.state_shape = get_state_shape(self.map.get_graph_size(), n_red, n_blue,
                                           obs_embed, obs_dir, obs_sight, obs_team)
        # agent obs size: [self: pos(6/27) + dir(4) + flags; teamB: pos + next_move + flags; teamR: 6 * (n_R - 1) or 27]
        self.observation_space = ObservationSpaces([spaces.Box(low=0, high=1, shape=(self.state_shape,), dtype=np.int8)
                                                    for _ in range(len(self.learning_agent))])
        self.invalid_masked = act_masked
        self.action_mask = []
        self.states = [[] for _ in range(len(self.learning_agent))]

    def reset(self, force=False):
        self.step_counter = 0
        if force:
            self.done_counter = 0
        self._reset_agents()
        self._update()
        return np.array(self.states, dtype=np.int8)

    def reset_step(self):
        self.step_counter = 0

    def step(self, n_actions):
        assert len(n_actions) == self.num_red, "[EnvError] Invalid action shape {}".format(n_actions)
        if self.logger is True:
            prev_obs = self.states
        self.step_counter += 1
        # take actions
        action_penalties = self._take_action_red(n_actions)
        self._take_action_blue()
        R_see_B, R_engage_B, B_see_R, B_engage_R, R_overlay = self._update()
        self.agent_interaction(R_engage_B, B_engage_R)
        # get rewards
        rewards = [0] * self.num_red
        rewards = self._step_rewards(rewards, action_penalties, R_engage_B, B_engage_R, R_overlay)
        # Done if agents lost all health points or reach max step
        dones = [bool(self.team_red[_r].get_health() <= 0 or self.step_counter >= self.max_step)
                 for _r in range(self.num_red)]
        # log action-states and step rewards
        if self.logger is True:
            _r = [["red:{}".format(_agent.get_id()), _agent.get_pos_dir(), _agent.get_encoding(), _agent.get_health()]
                  for _agent in self.team_red]
            _b = [["blue:{}".format(_agent.get_id()), _agent.get_pos_dir(), _agent.get_encoding(), _agent.get_health()]
                  for _agent in self.team_blue]
            save_log_2_file(self.logs, self.step_counter, self.done_counter, _r + _b,
                            prev_obs, n_actions, self.states, rewards)
        # add episodic reward and update done counts
        if all(done is True for done in dones):
            episode_rewards = self._episode_rewards()
            if self.logger is True:
                log_done_reward(self.logs, self.done_counter, episode_rewards)
            for _r in range(self.num_red):
                rewards[_r] += episode_rewards[_r]
            self.done_counter += 1
        return np.array(self.states, dtype=np.int8), rewards, dones, {}

    def _take_action_red(self, n_actions):
        action_penalty = [0] * self.num_red
        for agent_i, actions in enumerate(n_actions):
            # check input action if in range of the desired discrete action space
            assert self.action_space[agent_i].contains(actions), "{}: action out of range".format(actions)
            action_move, action_turn = actions

            # find all 1st ordered neighbors of the current node
            agent_encode = self.team_red[agent_i].agent_code
            prev_idx, list_neighbor, list_act = self.map.get_all_states_by_node(agent_encode)
            if action_move != 0 and action_move not in list_act:
                # if the action_mask turns on in the learning, invalid actions should not appear.
                if self.invalid_masked:
                    assert action_move, "[ActError] act{} {} {} mask{}".format(self.step_counter, action_move, prev_idx,
                                                                               self.action_mask[agent_i])
                # if the learning process doesn't have action masking, then invalid moves should be replaced by NOOP.
                else:
                    action_move = 0
                    action_penalty[agent_i] = INTERACT_LOOKUP["unmasked_invalid_action_penalty"]
            # just make turn when the move action is NOOP
            if action_move == 0:
                if "penalty_stay" in self.rewards["step_agent"]:
                    action_penalty[agent_i] += self.rewards["step_agent"]["penalty_stay"]
                if action_turn == 1:
                    self.team_red[agent_i].agent_dir = TURN_L[self.team_red[agent_i].agent_dir]
                elif action_turn == 2:
                    self.team_red[agent_i].agent_dir = TURN_R[self.team_red[agent_i].agent_dir]
            # make move and turn for the red agent
            elif action_move in list_act:
                self.team_red[agent_i].agent_node = list_neighbor[list_act.index(action_move)]
                self.team_red[agent_i].agent_code = self.map.get_name_by_index(self.team_red[agent_i].agent_node)
                if action_turn:
                    self.team_red[agent_i].agent_dir = TURN_L[action_move] if action_turn == 1 else TURN_R[action_move]
                else:
                    self.team_red[agent_i].agent_dir = action_move

        return action_penalty

    def _take_action_blue(self):
        for agent_i in range(self.num_blue):
            _route = self.team_blue[agent_i].get_route()
            _idx = self.step_counter % self.routes[_route].get_route_length()
            _node, _code, _dir = self.routes[_route].get_location_by_index(_idx)
            self.team_blue[agent_i].update_index(_idx, _node, _code, _dir)

    # update local states after all agents finished actions
    def _update(self):
        # generate binary matrices for pair-wised inSight and inRange indicators
        R_see_B = np.zeros((self.num_red, self.num_blue), dtype=np.bool_)
        R_engage_B = np.zeros((self.num_red, self.num_blue), dtype=np.bool_)
        B_see_R = np.zeros((self.num_blue, self.num_red), dtype=np.bool_)
        B_engage_R = np.zeros((self.num_blue, self.num_red), dtype=np.bool_)
        R_nodes = [0] * self.num_red
        R_overlay = [False] * self.num_red
        if self.invalid_masked:
            self.action_mask = [np.zeros(sum(tuple(self.action_space[_].nvec)), dtype=np.bool_)
                                for _ in range(len(self.learning_agent))]
        for _r in range(self.num_red):
            node_r, _ = self.team_red[_r].get_pos_dir()
            R_nodes[_r] = node_r
            for _b in range(self.num_blue):
                node_b, _ = self.team_blue[_b].get_pos_dir()
                R_see_B[_r, _b] = self.is_in_sight(node_r, node_b, self.team_red[_r].agent_dir)
                R_engage_B[_r, _b] = self.is_in_range(node_r, node_b, self.team_red[_r].agent_dir)
                B_see_R[_b, _r] = self.is_in_sight(node_b, node_r, self.team_blue[_b].agent_dir)
                B_engage_R[_b, _r] = self.is_in_range(node_b, node_r, self.team_blue[_b].agent_dir)
            # update action masking
            if self.invalid_masked:
                # self.action_mask[_r] = np.zeros(sum(tuple(self.action_space[_r].nvec)), dtype=np.bool_)
                mask_idx = self.learning_agent.index(self.team_red[_r].get_id())
                # masking invalid movements on the given node
                acts = set(ACTION_LOOKUP.keys())
                valid = set(self.map.get_actions_by_node(self.team_red[_r].get_encoding()) + [0])
                invalid = [act for act in acts if act not in valid]
                for masking in invalid:
                    self.action_mask[mask_idx][masking] = True

        # update overlap list for team red
        for _s in range(self.num_red - 1):
            for _t in range(_s + 1, self.num_red):
                if R_nodes[_s] == R_nodes[_t]:
                    R_overlay[_s] = True
                    R_overlay[_t] = True

        # update states for all agents in team red
        dir_placeholder_shape = (len(ACTION_LOOKUP) - 1)
        _obs_self_dir = self.configs["obs_dir"]
        _state_R_dir = []
        # get looking direction encodings for all red agents
        if _obs_self_dir:
            _state_R_dir = np.zeros((self.num_red, dir_placeholder_shape))
            for _r in range(self.num_red):
                _, _dir = self.team_red[_r].get_pos_dir()
                _state_R_dir[_r, (_dir - 1)] = 1

        # get next_move_dir encodings for all blue agents
        _state_B_next = np.zeros((self.num_blue, dir_placeholder_shape))
        for _b in range(self.num_blue):
            _route = self.team_blue[_b].get_route()
            _index = self.team_blue[_b].get_index()
            _dir = self.routes[_route].list_next[_index]
            _state_B_next[_b, (_dir - 1)] = 1

        ''' update state for each agent '''
        if self.configs["obs_embed"]:
            # condition: generate binary encoded position embedding for all agents
            _state_R_embed = []
            for agent_i in range(self.num_red):
                _state_R_embed.append(get_emb_from_name(self.team_red[agent_i].get_encoding()))
            _state_B_embed = []
            for agent_i in range(self.num_blue):
                _state_B_embed.append(get_emb_from_name(self.team_blue[agent_i].get_encoding()))

            # concatenate state_self + state_blues (+ state_reds)
            for _r in range(self.num_red):
                _state = []
                _state += _state_R_embed[_r]
                if _obs_self_dir:
                    _state += _state_R_dir[_r, :].tolist()
                if self.configs["obs_sight"]:
                    _state += R_see_B[_r, :].tolist()
                _state += R_engage_B[_r, :].tolist()

                # add team blue info
                for _b in range(self.num_blue):
                    _state += _state_B_embed[_b] + _state_B_next[_b, :].tolist()
                if self.configs["obs_sight"]:
                    _state += B_see_R[:, _r].tolist()
                _state += B_engage_R[:, _r].tolist()

                # add teammates info if True
                if self.configs["obs_team"]:
                    for agent in range(self.num_red):
                        if agent != _r:
                            _state += _state_R_embed[agent]
                self.states[_r] = _state
            # [Debug] test state shape
            # self.states = [[[(_ + self.step_counter) % 2] * self.state_shape] for _ in range(self.num_red)]
        else:
            # condition: multi-hot pos encodings for self, 'team_blue' and 'team_red' in the shape of len(G.all_nodes())
            pos_obs_size = self.map.get_graph_size()
            # concatenate state_self + state_blues (+ state_reds)
            for _r in range(self.num_red):
                # add self position one-hot embedding
                _state = [0] * pos_obs_size
                _state[R_nodes[_r] - 1] = 1
                # add self direction if True
                if _obs_self_dir:
                    _state += _state_R_dir[_r, :].tolist()
                # add self interaction indicators
                if self.configs["obs_sight"]:
                    _state += R_see_B[_r, :].tolist()
                _state += R_engage_B[_r, :].tolist()

                # add team blue info
                _state_B = [0] * pos_obs_size
                for _b in range(self.num_blue):
                    _node, _ = self.team_blue[_b].get_pos_dir()
                    _state_B[_node - 1] = 1
                    # add next move dir
                    _state_B += _state_B_next[_b, :].tolist()
                if self.configs["obs_sight"]:
                    _state_B += B_see_R[:, _r].tolist()
                _state_B += B_engage_R[:, _r].tolist()
                _state += _state_B

                # add teammates pos if True
                if self.configs["obs_team"]:
                    _state_R = [0] * pos_obs_size
                    for _agent in range(self.num_red):
                        if _agent != _r:
                            _state_R[R_nodes[_agent] - 1] = 1
                    _state += _state_R

                # update the local state attribute
                self.states[_r] = _state
            # [Debug] test state shape
            # self.states = [[[(_ + self.step_counter) % 2] * self.state_shape] for _ in range(self.num_red)]
        return R_see_B, R_engage_B, B_see_R, B_engage_R, R_overlay

    # update health points for all agents
    def agent_interaction(self, R_engage_B, B_engage_R):
        # update health and damage points for all agents
        for _b in range(self.num_blue):
            for _r in range(self.num_red):
                if R_engage_B[_r, _b]:
                    self.team_red[_r].damage_add(self.configs["damage_step"])
                    self.team_blue[_b].take_damage(self.configs["damage_step"])
                if B_engage_R[_b, _r]:
                    self.team_red[_r].take_damage(self.configs["damage_step"])
            # update end time for blue agents
            if self.team_blue[_b].get_end_step() > 0:
                continue
            blue_damage_taken = (self.configs["init_health"] - self.team_blue[_b].get_health())
            if blue_damage_taken >= self.configs["damage_threshold_blue"]:
                self.team_blue[_b].set_end_step(self.step_counter)

    def _step_rewards(self, rewards, penalties, R_engage_B, B_engage_R, R_overlay):
        for agent_r in range(self.num_red):
            rewards[agent_r] += penalties[agent_r] + get_step_overlay(R_overlay[agent_r], **self.rewards["step_agent"])
            for agent_b in range(self.num_blue):
                rewards[agent_r] += get_step_engage(r_engages_b=R_engage_B[agent_r, agent_b],
                                                    b_engages_r=B_engage_R[agent_b, agent_r],
                                                    team_switch=False, **self.rewards["step_agent"])
        return rewards

    def _episode_rewards(self):
        # gather final states for team blue
        rewards = [0] * self.num_red
        _health_b = [0] * self.num_blue
        _end_b = [0] * self.num_blue
        for _b in range(self.num_blue):
            _health_b[_b] = self.team_blue[_b].get_health()
            _end_b[_b] = self.team_blue[_b].get_end_step()

        _health_r = [0] * self.num_red
        _damage_r = [0] * self.num_red
        _health_full = self.configs["init_health"]
        _thres_r = self.configs["damage_threshold_red"]
        _thres_b = self.configs["damage_threshold_blue"]

        for agent_r in range(self.num_red):
            _health_r[agent_r] = self.team_red[agent_r].get_health()
            _damage_r[agent_r] = self.team_red[agent_r].damage_total()

            for agent_b in range(self.num_blue):
                rewards[agent_r] += get_episode_reward_agent(_health_full, _health_r[agent_r], _health_b[agent_b],
                                                             _thres_r, _thres_b, _damage_r[agent_r],
                                                             _end_b[agent_b], **self.rewards["done_agent"])

        _team_rewards = get_episode_reward_team(_health_r, _health_b, _health_full, _damage_r,
                                                _thres_r, _thres_b, _end_b) # , self.rewards["done_team"])
        for _i in range(self.num_red):
            rewards[_i] += _team_rewards
        return rewards

    def is_in_half(self, agent_idx, route_len):
        return not (agent_idx < (route_len // 2))

    def is_in_sight(self, s_idx, t_idx, s_dir):
        # check if there is an edge in the visibility FOV graph; if so, check if it is inside the sight range
        # no self-loop in the visibility graph for now, check if two agents are on the same node first
        if s_idx == t_idx:
            return True
        if self.map.g_vis.has_edge(s_idx, t_idx):
            _dis = self.map.get_edge_attr_vis_fov_by_idx(s_idx, t_idx, s_dir)
            # -1 indicates there is no visibility edge in the 's_dir' direction
            if _dis == -1:
                return False
            if INTERACT_LOOKUP["sight_range"] < 0 or (_dis < INTERACT_LOOKUP["sight_range"]):
                return True
        return False

    def is_in_range(self, s_idx, t_idx, s_dir):
        # check if there is an edge in visibility graph; if so, check if the distance is below the engaging range limit
        if s_idx == t_idx:
            return True
        if self.map.g_vis.has_edge(s_idx, t_idx):
            _dis = self.map.get_edge_attr_vis_fov_by_idx(s_idx, t_idx, s_dir)
            if _dis == -1:
                return False
            if _dis < INTERACT_LOOKUP["engage_range"]:
                return True
        return False

    def _init_env_config(self, **kwargs):
        # set default env config values if not specified in outer configs
        _config_args = ["init_red", "init_blue", "damage_step", "damage_threshold_red", "damage_threshold_blue"]
        INIT_REGION = {"L": 1, "R": 0}
        self.rewards["step_agent"] = {}
        self.rewards["done_agent"] = {}
        _reward_agent_step = ["reward_step_RB", "reward_step_BR", "reward_step_RR", "penalty_stay"]
        _reward_agent_done = ["reward_episode_lookup", "reward_faster_lookup"]
        # Designed for eval. default path is None. If None, no log files will be generated during step run.
        _log_keys = ["on", "save", "plot", "path", "prefix", "overview", "verbose"]
        # log defaults
        LOG_LOOKUP = {"prefix": "log_", "local_path": "logs/", "save": True,
                      "verbose": False, "plot": False, "overview": "reward_episodes.txt"}
        for item in kwargs:
            if item in _config_args:
                self.configs[item] = kwargs[item]
            elif item in _reward_agent_step:
                self.rewards["step_agent"][item] = kwargs[item]
            elif item in _reward_agent_done:
                self.rewards["done_agent"][item] = kwargs[item]
            elif item in _log_keys:
                self.logs[item] = kwargs[LOG_LOOKUP["prefix"]+item]
            # else:
            #     pass

        # set default configs if not defined
        for key in _config_args:
            if key in self.configs:
                continue
            if key == "init_red":
                self.configs[key] = [{"learn": True, "pos": None, "dir": None} for _ in range(self.num_red)]
            elif key == "init_blue":
                self.configs[key] = [{"learn": False, "route": "0", "idx": 0} for _ in range(self.num_blue)]
            elif key == "damage_step":
                self.configs[key] = INTERACT_LOOKUP["engage_damage"]
            elif key == "damage_threshold_red":
                self.configs[key] = INTERACT_LOOKUP["damage_maximum"]
            elif key == "damage_threshold_blue":
                # grant blue agents a higher damage threshold when more reds on the map
                self.configs[key] = INTERACT_LOOKUP["damage_maximum"] * self.num_red
            else:
                print("[Env] Invalid config arg:{}".format(key))

        # check init_red init configs, must have attribute "pos":[str/tuple] for resetting agents
        assert len(self.configs["init_red"]) == self.num_red, "Invalid init_red:{}".format(self.configs["init_red"])
        for idx, _config in enumerate(self.configs["init_red"]):
            # set up default init looking direction "dir":[int]
            if "dir" not in _config:
                self.configs["init_red"][idx]["dir"] = None
            # parse "pos" indicator and validate range
            _pos = _config["pos"]
            if _pos is None:
                continue
            if isinstance(_pos, tuple):
                assert check_pos_abs_range(_pos), "Pos tuple: {} out of range".format(_pos)
            elif isinstance(_pos, str):
                assert _pos in INIT_REGION, "Invalid init region:\'{}\' (not in {})".format(_pos, INIT_REGION.keys())
                self.configs["init_red"][idx]["pos"] = INIT_REGION[_pos]


        # check init_blue, must have attribute "route":[str] used in loading graph files. default: '0'
        assert len(self.configs["init_blue"]) == self.num_blue, "Invalid init_blue:{}".format(self.configs["init_blue"])
        _route_list = []
        for idx, _config in enumerate(self.configs["init_blue"]):
            _route_list.append(_config["route"])
            # set up default spwan index on the route "idx":[int]
            if "idx" not in _config:
                self.configs["init_red"][idx]["idx"] = 0
        # get all unique routes. (blue agents might share patrol route)
        self.configs["route_lookup"] = list(set(_route_list))

        # turn on logger if True
        if "on" in self.logs:
            self.logger = self.logs["on"]
            if self.logger is True:
                self.logs["root_path"] = self.configs["env_path"]
                for item in _log_keys[1:]:
                    if item not in self.logs:
                        self.logs[item] = LOG_LOOKUP[item]
        return True

    # load local graphs and generate routes [!!] executable after _init_env_config
    def _load_map_data(self):
        # load graphs
        self.map, self.routes = load_graph_files(env_path=self.configs["env_path"],
                                                 map_lookup=self.configs["map_id"],
                                                 route_lookup=self.configs["route_lookup"],
                                                 is_pickle_graph=self.configs["load_pickle"])
        # generate pos & dirs for patrol routes
        for _route in range(len(self.routes)):
            route_len = self.routes[_route].get_route_length()
            self.routes[_route].list_node = [0] * route_len
            for _idx in range(route_len):
                self.routes[_route].list_node[_idx] = self.map.get_index_by_name(self.routes[_route].list_code[_idx])
            list_dir = [0] * (route_len - 1)
            for _idx in range(route_len - 1):
                list_dir[_idx] = self.map.get_edge_attr_acs_by_idx(self.routes[_route].list_node[_idx],
                                                                   self.routes[_route].list_node[_idx + 1])
            off_dir = self.map.get_edge_attr_acs_by_idx(self.routes[_route].list_node[-1],
                                                        self.routes[_route].list_node[0])
            # verbose direction info for fast retrieval
            self.routes[_route].list_move = [off_dir] + list_dir
            self.routes[_route].list_next = list_dir + [off_dir]

    # initialize team red and team blue agents [!!] executable after _init_env_config
    def _init_agents(self):
        health = self.configs["init_health"]
        for idx in range(self.num_red):
            r_uid = idx
            learn = self.configs["init_red"][idx]["learn"] if "learn" in self.configs["init_red"][idx] else True
            if learn is True:
                self.learning_agent.append(r_uid)
            self.team_red.append(AgentRed(_uid=r_uid, _health=health, _learn=learn))

        for idx in range(self.num_blue):
            b_uid = (idx + self.num_red)
            learn = self.configs["init_blue"][idx]["learn"] if "learn" in self.configs["init_blue"][idx] else False
            if learn is True:
                self.learning_agent.append(b_uid)
            b_route = self.configs["route_lookup"].index(self.configs["init_blue"][idx]["route"])
            b_index = self.configs["init_blue"][idx]["idx"]
            self.team_blue.append(AgentBlue(_uid=b_uid, _health=health, _learn=learn, _route=b_route, _index=b_index))

    # reset agents to init status for each new episode
    def _reset_agents(self):
        health = self.configs["init_health"]
        for idx in range(self.num_red):
            # parsing init position config
            _flag = self.configs["init_red"][idx]["pos"]
            _args = [None, 0]
            if isinstance(_flag, int):
                _args[1] = _flag
            elif isinstance(_flag, tuple):
                _args[0] = _flag
            r_code = get_default_red_encoding(_args[0], _args[1])
            r_node = self.map.get_index_by_name(r_code)
            r_dir = get_default_dir(self.configs["init_red"][idx]["dir"])
            self.team_red[idx].reset(_node=r_node, _code=r_code, _dir=r_dir, _health=health)

        for idx in range(self.num_blue):
            b_route = self.team_blue[idx].get_route()
            b_index = int(self.configs["init_blue"][idx]["idx"])
            b_node, b_code, b_dir = self.routes[b_route].get_location_by_index(b_index)
            self.team_blue[idx].reset(_node=b_node, _code=b_code, _dir=b_dir, _health=health, _index=b_index, _end=-1)
            if self.team_blue[idx].is_learning():
                self.learning_agent.append(self.team_blue[idx].agent_id)

    def render(self, mode='human'):
        pass

    # delete all local instances
    def close(self):
        if self.team_red is not None:
            del self.team_red
        if self.team_blue is not None:
            del self.team_blue
        if self.routes is not None:
            del self.routes
        if self.map is not None:
            del self.map


# update agent position. return value: str of binary embedding code
def get_default_red_encoding(red_pos=None, red_region=0) -> str:
    # providing the coordinates of red position (row, col) or sampling from init spawn position pools
    if isinstance(red_pos, tuple):
        return get_node_name_from_pos_abs(red_pos)
    else:
        pos_red_pool = INIT_POS_LOOKUP[red_region % len(INIT_POS_LOOKUP)]
        idx = randint(0, len(pos_red_pool) - 1)
        R_pos = pos_red_pool[idx]
    return get_node_name_from_pos_abs(R_pos)


# update agent looking direction. return value: int direction indicator
def get_default_dir(default_dir=None) -> int:
    if default_dir is not None:
        return default_dir
    return randint(min(ACTION_LOOKUP.keys()) + 1, max(ACTION_LOOKUP.keys()))


# get default observation shape per agent
def get_state_shape(n_G, n_R, n_B, obs_embed=False, obs_self_dir=True, obs_sight=True, obs_team=True) -> int:
    (bit_row, bit_col), _ = get_pos_norms()
    bit_embed = bit_row + bit_col
    look_dir = len(ACTION_LOOKUP) - 1
    # agent states for local obs and global view of opponents and teammates
    state_self = (bit_embed if obs_embed else n_G) + obs_self_dir * look_dir + (obs_sight + 1) * n_B
    state_B = (((bit_embed + look_dir) * n_B) if obs_embed else (n_G + look_dir)) + (1 + obs_sight) * n_B
    state_R = obs_team * (bit_embed * (n_R - 1) if obs_embed else n_G)
    return state_self + state_B + state_R


class AgentRed(MAgent):
    def __init__(self, _uid=0, _node=0, _code=None, _dir=0, _health=5, _learn=True):
        super().__init__(_uid, _node, _code, _dir, _health, _learn)
        self.total_damage = 0

    def reset(self, _node, _code, _dir, _health):
        super().reset(_node=_node, _code=_code, _dir=_dir, _health=_health)
        self.total_damage = 0

    def damage_total(self):
        return self.total_damage

    def damage_add(self, points=1):
        self.total_damage += points

    def close(self):
        pass


class AgentBlue(MAgent):
    def __init__(self, _uid=0, _node=0, _code=None, _dir=0, _health=5, _learn=False, _route=None, _index=None):
        super().__init__(_uid, _node, _code, _dir, _health, _learn)
        self.route_ptr = _route
        self.route_idx = _index
        self.end_step = -1

    def reset(self, _node, _code, _dir, _health, _index=0, _end=-1):
        super().reset(_node=_node, _code=_code, _dir=_dir, _health=_health)
        self.route_idx = _index
        self.end_step = _end

    def get_route(self):
        return self.route_ptr

    def get_index(self):
        return self.route_idx

    def set_end_step(self, step):
        self.end_step = step

    def get_end_step(self):
        return self.end_step

    def update_index(self, _idx, _node, _code, _dir):
        self.route_idx = _idx
        super().set_location(_node, _code, _dir)

    def close(self):
        pass
