import numpy as np
from random import randint

import gym
from gym import spaces

from ..utils.multiagent_space import ActionSpaces, ObservationSpaces
from ..utils.multiagent_agent_base import MAgent
from ...data.file_manager import load_graph_files

from .maps.skirmish_graph import MapInfo, RouteInfo
from .maps.configs import ACTION_LOOKUP, ACTION_TURN_LOOKUP, TURN_L, TURN_R, INTERACT_LOOKUP, INIT_POS_LOOKUP
from .maps.data_helper import get_node_name_from_pos_abs, get_node_pos_from_name_abs, get_pos_norms, get_emb_from_name
from .rewards.rewards_simple import get_step_reward, get_step_overlay, get_episode_reward_agent, get_episode_reward_team


class Figure8Squad(gym.Env):
    def __init__(self, env_path='./', env_map='S', load_pkl=True, n_red=2, init_red=None, n_blue=1, init_blue=None,
                 init_health=100, d_step=INTERACT_LOOKUP["engage_damage"], d_agent=INTERACT_LOOKUP["damage_threshold"],
                 max_step=40, act_masked=True, obs_embed=False, obs_dir=True, obs_team=True, obs_sight=True, **args):
        self.max_step = max_step
        self.step_counter = 0
        # setup env configs
        if init_red is None:
            init_red = [{"pos": None, "dir": None} for _ in range(n_red)]
        if init_blue is None:
            init_blue = [{"route": "0", "idx": 0} for _ in range(n_blue)]
        self.list_route = list(set([_d["route"] for _d in init_blue]))
        self.configs = {"env_path": env_path, "map_lookup": env_map, "load_pickle": load_pkl,
                        "init_red": init_red, "init_blue": init_blue, "init_health": init_health,
                        "damage_step": d_step, "damage_thres": d_agent, "invalid_masked": act_masked,
                        "obs_dir": obs_dir, "obs_embed": obs_embed, "obs_team": obs_team, "obs_sight": obs_sight}
        self.rewards = {}
        self.logs = {}
        self.set_outer_configs(self, **args)
        # init map: load connectivity & visibility graphs and patrol routes
        self.map = MapInfo()
        self.routes = []
        self._load_map_data(self.list_route)

        # init agents
        self.num_red = n_red
        self.num_blue = n_blue
        # agent instances
        self.team_red = [AgentRed(_uid=_, _health=init_health) for _ in range(n_red)]
        self.team_blue = [AgentBlue(_uid=(_ + n_red), _health=init_health) for _ in range(n_blue)]

        # init spaces
        self.action_space = ActionSpaces([spaces.MultiDiscrete([len(ACTION_LOOKUP), len(ACTION_TURN_LOOKUP)])
                                          for _ in range(n_red)])
        # agent obs size: [self: pos(6/27) + dir(4) + flags; teamB: pos + next_move + flags; teamR: 6 * (n_R - 1) or 27]
        self.state_shape = get_state_shape(self.map.get_graph_size(), n_red, n_blue,
                                           obs_embed, obs_dir, obs_sight, obs_team)
        self.observation_space = ObservationSpaces([spaces.Box(low=0, high=1, shape=(self.state_shape,), dtype=np.int8)
                                                    for _ in range(n_red)])
        self.action_mask = []
        if act_masked:
            self.action_mask = [np.empty((len(ACTION_LOOKUP) + len(ACTION_TURN_LOOKUP)), dtype=np.bool_)
                                for _ in range(n_red)]
        self.states = [[] for _ in range(n_red)]
        self.done_counter = 0

    def reset(self):
        self.step_counter = 0
        # self.done_counter = 0
        self._reset_agents()
        self._update()
        return np.array(self.states, dtype=np.int8)

    def step(self, n_actions):
        assert len(n_actions) == self.num_red
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
        dones = [bool(self.team_red[_r].get_health() <= 0 or self.step_counter >= self.max_step) for _r in
                 range(self.num_red)]
        if all(done is True for done in dones):
            rewards = self._episode_rewards(rewards)
        # logger for eval and visuals
        return np.array(self.states, dtype=np.int8), rewards, dones, {}

    def _take_action_red(self, n_actions):
        action_penalty = [0] * self.num_red
        for agent_i, actions in enumerate(n_actions):
            action_move, action_turn = actions
            # find all 1st ordered neighbors of the current node
            agent_encode = self.team_red[agent_i].agent_code
            prev_idx, list_neighbor, list_act = self.map.get_all_states_by_node(agent_encode)
            if action_move != 0 and action_move not in list_act:
                # if the action_mask turns on in the learning, invalid actions should not appear.
                if self.configs["invalid_masked"]:
                    assert action_move, "[ActError] act{} {} {} mask{}".format(self.step_counter, action_move, prev_idx,
                                                                               self.action_mask[agent_i],
                                                                               get_node_pos_from_name_abs(agent_encode))
                # if the learning process doesn't have action masking, then invalid moves should be replaced by NOOP.
                else:
                    action_move = 0
                    action_penalty[agent_i] = INTERACT_LOOKUP["unmasked_invalid_action_penalty"]
            # just make turn when the move action is NOOP
            if action_move == 0:
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
            if self.configs["invalid_masked"]:
                self.action_mask[_r] = np.zeros((len(ACTION_LOOKUP) + len(ACTION_TURN_LOOKUP)), dtype=np.bool_)
                # masking invalid movements on the given node
                acts = set(ACTION_LOOKUP.keys())
                valid = set(self.map.get_actions_by_node(self.team_red[_r].get_encoding()) + [0])
                invalid = [act for act in acts if act not in valid]
                for masking in invalid:
                    self.action_mask[_r][masking] = True

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

        # update state for each agent
        if self.configs["obs_embed"]:
            # generate binary position encodings for all agents
            _state_R_embed = []
            for agent_i in range(self.num_red):
                _state_R_embed.append(get_emb_from_name(self.team_red[agent_i].get_encoding()))
            _state_B_embed = []
            for agent_i in range(self.num_blue):
                _state_B_embed.append(get_emb_from_name(self.team_blue[agent_i].get_encoding()))

            # concatenate state_self + state_blue + state_red
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
            # generate multi-hot encodings for self, 'team_blue' and 'team_red' in the shape of len(G.all_nodes())
            pos_obs_size = self.map.get_graph_size()
            # concatenate state_self + state_blue + state_red
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
        # end game condition for blue agent: Thres * n_Red
        _threshold_blue = self.configs["damage_thres"] * self.num_red  # TODO: modify the condition for more red agents
        for _b in range(self.num_blue):
            for _r in range(self.num_red):
                if R_engage_B[_r, _b]:
                    self.team_red[_r].damage_add(self.configs["damage_step"])
                    self.team_blue[_b].take_damage(self.configs["damage_step"])
                if B_engage_R[_b, _r]:
                    self.team_red[_r].take_damage(self.configs["damage_step"])
            # update end time for blue agents
            _damage_taken_blue = self.configs["init_health"] - self.team_blue[_b].get_health()
            if _damage_taken_blue >= _threshold_blue and self.team_blue[_b].get_end_step() < 0:
                self.team_blue[_b].set_end_step(self.step_counter)

    def _step_rewards(self, rewards, penalties, R_engage_B, B_engage_R, R_overlay):
        for agent_r in range(self.num_red):
            rewards[agent_r] += penalties[agent_r] + get_step_overlay(R_overlay[agent_r])
            for agent_b in range(self.num_blue):
                rewards[agent_r] += get_step_reward(R_engage_B[agent_r, agent_b], B_engage_R[agent_b, agent_r])
        return rewards

    def _episode_rewards(self, rewards, penalties=None):
        # gather final states for team blue
        _health_b = [0] * self.num_blue
        _end_b = [0] * self.num_blue
        for _b in range(self.num_blue):
            _health_b[_b] = self.team_blue[_b].get_health()
            _end_b[_b] = self.team_blue[_b].get_end_step()

        _health_r = [0] * self.num_red
        _damage_r = [0] * self.num_red
        for agent_r in range(self.num_red):
            _health_r[agent_r] = self.team_red[agent_r].get_health()
            _damage_r[agent_r] = self.team_red[agent_r].damage_total()

            for agent_b in range(self.num_blue):
                rewards[agent_r] += get_episode_reward_agent(_health_r[agent_r], _health_b[agent_b],
                                                             self.configs["init_health"],
                                                             self.configs["damage_thres"] * self.num_blue,
                                                             self.configs["damage_thres"] * self.num_red,
                                                             _damage_r[agent_r], _end_b[agent_b])

        # _team_rewards = get_episode_reward_team(_health_r, _health_b, self.configs["init_health"], _damage_r, _end_b,
        #                                         self.configs["damage_thres"] * self.num_blue,
        #                                         self.configs["damage_thres"] * self.num_red)
        # for _i in range(self.num_red):
        #     rewards[_i] += _team_rewards
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

    def _reset_agents(self):
        for idx in range(self.num_red):
            _code = get_default_red_encoding(self.configs["init_red"][idx]["pos"], idx)
            _node = self.map.get_index_by_name(_code)
            _dir = get_default_dir(self.configs["init_red"][idx]["dir"])
            self.team_red[idx].reset(_node, _code, _dir, self.configs["init_health"])

        for idx in range(self.num_blue):
            _route = self.list_route.index(self.configs["init_blue"][idx]["route"])
            _index = int(self.configs["init_blue"][idx]["idx"])
            _node, _code, _dir = self.routes[_route].get_location_by_index(_index)
            self.team_blue[idx].reset(_node, _code, _dir, self.configs["init_health"], _route, _index)

    def _load_map_data(self, routes):
        # load graphs
        self.map, self.routes = load_graph_files(env_path=self.configs["env_path"],
                                                 map_lookup=self.configs["map_lookup"],
                                                 route_lookup=routes,
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

    def set_outer_configs(self, **args):
        # Designed for eval. default path is None. If None, no log files will be generated during step run.
        LOG_LOOKUP = {
            "log_on": False,
            "log_path": "sigma_graph/logs/",
            "log_prefix": "log_",
            "log_overview": "run_dones.txt",
        }
        # TODO: define keys
        list_config_keys = []
        list_reward_keys = ["reward_done_lookup", "reward_R_2_B", "reward_R_2_B", "reward_R_2_R", "reward_B_2_R",
                            "reward_fast", "step_pivot", "step_decay"]
        list_log_keys = ["log_on", "log_path", "log_overview", "log_prefix"]
        for item in args.keys():
            if item in list_config_keys:
                self.configs[item] = args[item]
            elif item in list_reward_keys:
                self.rewards[item] = args[item]
            elif item in list_log_keys:
                self.logs[item] = args[item]
        return True

    def render(self, mode='human'):
        pass

    def close(self):
        if self.team_red is not None:
            del self.team_red
        if self.team_blue is not None:
            del self.team_blue
        if self.routes is not None:
            del self.routes
        if self.map is not None:
            del self.map


# get red agent position for map initialization
def get_default_red_encoding(red_pos=None, _region=0) -> str:
    # providing the coordinates of red position (row, col) or sampling from init spawn position pools
    if red_pos is not None:
        return get_node_name_from_pos_abs(red_pos)
    else:
        pos_red_pool = INIT_POS_LOOKUP[_region % len(INIT_POS_LOOKUP)]
        idx = randint(0, len(pos_red_pool) - 1)
        R_pos = pos_red_pool[idx]
    return get_node_name_from_pos_abs(R_pos)


def get_default_dir(red_dir=None) -> int:
    if red_dir is not None:
        return red_dir
    return randint(min(ACTION_LOOKUP.keys()) + 1, max(ACTION_LOOKUP.keys()))


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
    def __init__(self, _uid=0, _node=0, _code=None, _dir=0, _health=5, _learn=False):
        super().__init__(_uid, _node, _code, _dir, _health, _learn)
        self.route_ptr = None
        self.route_idx = None
        self.end_step = -1

    def reset(self, _node, _code, _dir, _health, _route=0, _idx=0, _end=-1):
        super().reset(_node=_node, _code=_code, _dir=_dir, _health=_health)
        self.route_ptr = _route
        self.route_idx = _idx
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
