import gym
import numpy as np
from random import randrange, uniform

from graph_scout.envs.utils.agent.agent_cooperative import AgentCoop
from graph_scout.envs.utils.agent.agent_heuristic import AgentHeur
from graph_scout.envs.base.action_lookup import ActionBranched as actsEval


class ScoutMissionStd(gym.Env):
    def __init__(self, **kwargs):
        # 0 setup general elements and containers
        self.configs = {}
        self.agents = None
        self.states = None
        self.map = None

        self.step_counter = 0
        self.done_counter = 0

        # 1 init environment configuration arguments
        # 1.1 init all local/default configs and parse additional arguments
        self._init_env_config(**kwargs)
        # 1.2 load terrain related info (load parsed connectivity & visibility graphs)
        self._load_map_graph()
        # 1.3 generate agent & state instances
        self._init_agent_state()

        # 2 init Multi-branched action gym space & flattened observation gym space
        from gym import spaces
        self.action_space = [spaces.MultiDiscrete(self.acts.shape())] * self.states.num
        self.observation_space = [spaces.Box(low=0., high=1., shape=(self.states.shape*2,))] * self.states.num

    def reset(self, force=False, **kwargs):
        self.step_counter = 0
        if force:
            self.done_counter = 0
        self.agents.reset()
        self.states.reset()
        self.update()
        self.update_observation()
        return self.states.obs_full

    def reset_step_count(self):
        self.step_counter = 0

    def step(self, n_actions, force_stop=False, **kwargs):
        self.step_counter += 1

        # 1. manage actions
        _invalid = self.validate_actions(n_actions)
        self.prep_other_agent_state()

        # 2. update states
        self.update()
        # mini step interactions
        self.agent_interactions()
        self.update_observation()

        # 3. calculate rewards
        # step rewards: stored @self.states.rewards[:, n_step]
        if self.rew_cfg["step"]["rew_step_on"]:
            self.get_step_rewards(_invalid)
        n_rewards = self.states.get_reward_list(self.step_counter)
        # True: if an agent loses all health points (or @max_step)
        n_done = self.get_done_list(force_stop)
        if all(n_done):
            # update global done counts
            self.done_counter += 1
            # add episodic rewards; stored @self.states.rewards[:, 0]
            if self.rew_cfg["episode"]["rew_ep_on"]:
                _ep_rewards = self.get_episode_rewards()
                for _id in range(len(n_rewards)):
                    n_rewards[_id] += _ep_rewards[_id]
        return self.states.obs_full, n_rewards, n_done, {}

    # 1.1 manage actions for observing agents
    def validate_actions(self, step_actions):
        action_penalty = [0] * self.states.num
        assert self.states.num == len(step_actions), f"[GSMEnv][Step] Unexpected action shape: {step_actions}"

        for _index in range(self.states.num):
            actions = step_actions[_index]
            # Check input action if in range of the desired discrete action space
            # assert self.action_space[_index].contains(actions), f"[GSMEnv][Step] Action out of range: {actions}"
            _id = self.agents.ids_ob[_index]
            _agent = self.agents.gid[_id]

            # Skip actions for death agents
            if _agent.death:
                continue
            action_move, action_look, action_body = actions
            # Find all first order neighbors of the current node
            _node = _agent.at_node
            dict_move_node = self.map.get_Gmove_action_node_dict(_node)
            # Validate actions with masks
            if action_move and (action_move not in dict_move_node):
                # If the action_mask turns on in learning, invalid actions should not appear.
                if self.invalid_masked:
                    assert f"[GSMEnv][Step] action:{action_move} node:{_node} masking:{self.action_mask[_index]}"
                # If action masking doesn't applied, then the invalid Move should be replaced by 0:"NOOP".
                else:
                    action_move = 0
                    action_penalty[_index] = self.configs["penalty_invalid"]

            # Update state tuple(loc, dir, pos) after executing actions
            if action_move in dict_move_node:
                _node = dict_move_node[action_move]
            dir_anew = self.acts.look[action_look]
            pos_anew = action_body
            # Fill up engage matrix for later use.
            self.engage_mat[_id, -3:] = [dir_anew, pos_anew, _node]
            # Agent 'dir' & 'pos' are updated immediately; Do NOT update 'at_node' at this stage.
            _agent.set_acts(action_move, dir_anew, pos_anew)
            _agent.step_reset()
        return action_penalty

    # 1.2 manage actions for all other agents
    def prep_other_agent_state(self):
        """ # TODO (lv1) only support heuristic_blue + lr_red now
        # get all other agents' next states before executing mini-steps
        # decision trees for DT agents
        # behavior branches: {[0,1]:"FORWARD", [2,3,4]:"ASSIST", 5:"RETREAT"}
        """
        # DIR POS target selection
        for a_src in self.agents.ids_dt:
            # Skip death agent
            _agent = self.agents.gid[a_src]
            if _agent.death:
                continue
            # If current target_agent is None, set new target if possible
            src_node = self.engage_mat[a_src, a_src]
            _dir, _pos = self.engage_mat[a_src, -3:-1]
            target_agent_id = _agent.target_agent
            if target_agent_id < 0:
                # {target_agent_id: distance}
                tar_dist = {}
                # store all possible {target_aid: dist}
                for a_tar in self.agents.ids_R:
                    tar_node = self.engage_mat[a_tar, a_tar]
                    if self.map.g_view.has_edge(src_node, tar_node):
                        tar_dist[a_tar] = self.map.get_Gview_edge_attr_dist(src_node, tar_node)
                # If opposite agents are in sight
                if any(tar_dist):
                    _ids = list(tar_dist.keys())
                    _dist = list(tar_dist.values())
                    # Select the nearest target
                    target_dist = min(_dist)
                    a_tar = _ids[_dist.index(target_dist)]
                    _agent.target_agent = a_tar
                    tar_node = self.engage_mat[a_tar, a_tar]
                    _dir = self.map.get_Gview_edge_attr_dir(src_node, tar_node, 0)
                    _pos = self.map.get_Gview_edge_attr_pos(src_node, tar_node, 0)
                    zone_id = self._get_zone_by_dist(target_dist)
                    _agent.target_zone = zone_id
                else:
                    # random acts if no target in range
                    _dir = randrange(self.acts.n_look)  # 1 for always looking North
                    _pos = 0  # randrange(self.acts.n_body) always 0:"Stand"
            # Keep current target and update zone token
            else:
                tar_node = self.engage_mat[target_agent_id, target_agent_id]
                if self.map.g_view.has_edge(src_node, tar_node):
                    # maintain current target if it is still in sight
                    _dir = self.map.get_Gview_edge_attr_dir(src_node, tar_node, 0)
                    zone_id = self._get_zone_by_dist(self.map.get_Gview_edge_attr_dist(src_node, tar_node))
                    _agent.target_zone = zone_id
                else:
                    # reset target to default
                    self.dt_buffer[a_src] -= 1
                    _agent.target_zone = 0
                    if self.dt_buffer[a_src] == 0:
                        _agent.target_agent = -1
                        self.dt_buffer[a_src] = self.configs["buffer_count"]
            _agent.direction = _dir
            self.engage_mat[a_src, -3] = _dir
            # Update agent.posture after all mini-steps using mat[agent, 'pos']
            self.engage_mat[a_src, -2] = _pos

        # MOVE
        if self.step_counter < self.configs["num_hibernate"]:
            return True
        for _index, a_src in enumerate(self.agents.ids_dt):
            _agent = self.agents.gid[a_src]
            # Skip death agent
            if _agent.death:
                continue
            # Move in macro step
            if _agent.slow_mode:
                self.engage_mat[a_src, -1] = _agent.move_slow_mode_prep()
            else:
                self.engage_mat[a_src, -1] = _agent.move_en_route_prep()

            # change action branch
            cur_branch = self.assist_mat[_index, _index]
            #tar_branch = self._get_val_from_lists(_agent.health / _agent.health_max, **self.configs["branch_dict"])
            tar_branch = 1
            # default forward
            if tar_branch <= cur_branch:
                continue
            if tar_branch > 4:
                # retreat
                _agent.target_branch = tar_branch
                self.assist_mat[_index, _index] = tar_branch
                a_tar = self.assist_mat[_index, -1]
                self.assist_mat[a_tar, _index] = 0
                # [TBD] reset path
                self.assist_mat[a_tar, a_tar] = 0
            elif tar_branch > 1:
                # assist
                _agent.target_branch = tar_branch
                self.assist_mat[_index, _index] = tar_branch
                for a_tar in range(len(self.agents.ids_dt)):
                    # teammate in another branch
                    if self.assist_mat[a_tar, a_tar]:
                        continue
                    # [TBD] new path
                    self.assist_mat[_index, -1] = a_tar
                    self.assist_mat[a_tar, a_tar] = 4
                    self.assist_mat[a_tar, _index] = tar_branch
        return False

    # 2.1 update local states after all agents get action validation done
    def update(self):
        # Update engage matrix
        self._reset_engage_matrix()
        self._update_engage_matrix()
        # generate new invalid action masking for all observing agents
        if self.invalid_masked:
            # update action masking
            self.action_mask[:] = False
            for _index, a_id in enumerate(self.agents.ids_ob):
                # masking invalid move actions on the given node
                acts = self.act_move_set  # set(self.acts.move.keys())
                valid = set(self.map.get_Gmove_all_action(self.agents.gid[a_id].at_node) + [0])
                invalid = [_act for _act in acts if _act not in valid]
                self.action_mask[_index][invalid] = True
        return False

    # 2.1.1 engage matrix updates
    def _update_engage_matrix(self):
        # TBD(lv1) only support heuristic_blue + lr_red
        # each agent only have (at most) ONE active target agent
        # ===> one zone token per row
        for _r in self.agents.ids_R:
            _u = self.engage_mat[_r, _r]
            max_zone, max_id = [0, 0]
            for _b in self.agents.ids_B:
                _v = self.engage_mat[_b, _b]
                _zone = self._get_zone_token(_u, _v)
                if _zone and _zone > max_zone:
                    max_zone = _zone
                    max_id = _b
            if max_zone:
                self.engage_mat[_r, max_id] = max_zone
        # update engage matrix for heuristic agents
        for _b in self.agents.ids_B:
            _agent = self.agents.gid[_b]
            _r = _agent.target_agent
            if _r < 0 or _agent.target_branch == 5:
                continue
            if _agent.target_zone:
                self.engage_mat[_b, _r] = _agent.target_zone
        return False

    # 2.1.2 engage matrix resets
    def _reset_engage_matrix(self):
        # engage_mat token clean up -> reset to 0 for the next step
        for _r in self.agents.ids_R:
            for _b in self.agents.ids_B:
                self.engage_mat[_r, _b] = 0
                self.engage_mat[_b, _r] = 0
        return False

    # 2.2 update health points for all agents
    def agent_interactions(self):
        # TBD(lv1) only support heuristic_blue + lr_red now
        # update health and damage points for all agents
        _field_range = self.configs["field_boundary_node"]
        _field_damage = self.configs["damage_field"]
        _step_damage = self.configs["damage_single"]
        n_mini_step = self.configs["num_sub_step"]
        mid_step = int(n_mini_step / 2)  # use self.configs["num_sub_node"] for more reference sub-nodes

        for sub_step in range(n_mini_step):
            # change anchor node from move_src to move_tar after half mini_steps
            if sub_step == mid_step:
                # make the real MOVE action (update agent.at_node info) after a certain mini-steps
                self._update_sub_locations()

            # check engage matrix to get the engagement zone token for each red
            for _r in self.agents.ids_R:
                _agent = self.agents.gid[_r]
                for _b in self.agents.ids_B:
                    # check reed_blue engage flags
                    token_r_b = self.engage_mat[_r, _b]
                    if token_r_b:
                        _damage = self._get_prob_by_src_tar(_r, _b, token_r_b)
                        if _damage:
                            # shooting successful
                            _agent.damage_given(_step_damage)
                            self.agents.gid[_b].damage_taken(_step_damage)
                        else:
                            # even though no shooting, marks disturbing
                            _agent.disturbing()
                        # If an agent only has a single target at a given time step -> early break
                        break
                # get field damage if in MG's range
                if _agent.at_node < _field_range:
                    _agent.damage_taken(_field_damage)

            # check engage matrix for each blue
            for _b in self.agents.ids_B:
                for _r in self.agents.ids_R:
                    # check blue_red engage flags
                    token_b_r = self.engage_mat[_b, _r]
                    if token_b_r:
                        if self._get_prob_by_src_tar(_b, _r, token_b_r):
                            self.agents.gid[_b].damage_given(_step_damage)
                            self.agents.gid[_r].damage_taken(_step_damage)
                        # single target -> early break
                        break

        for _b in self.agents.ids_B:
            _agent = self.agents.gid[_b]
            # update action branch
            if _agent.death:
                continue
            # update blues' postures
            _pos_now = _agent.posture
            _pos_new = self.engage_mat[_b, -2]
            if _pos_now:
                if _agent.if_at_main_nodes():
                    # pos 1 -> 0
                    _agent.change_speed_fast()
                    self.engage_mat[_b, -2] = 0
            elif _pos_new:
                # pos 0 -> 1
                _agent.change_speed_slow()
        return False

    # 2.2.1 change anchoring nodes during mini-steps
    def _update_sub_locations(self):
        self._reset_engage_matrix()
        for _id in range(self.agents.n_all):
            node_anew = self.engage_mat[_id, -1]
            if self.engage_mat[_id, _id] != node_anew:
                self.engage_mat[_id, _id] = node_anew
                self.agents.gid[_id].at_node = node_anew
        self._update_engage_matrix()

    # 2.2.2 get the binary token for pair-wise engagement
    def _get_prob_by_src_tar(self, u_id, v_id, zone_id) -> bool:
        # input: int zone_id > 0
        if zone_id == 4:
            # overlapping agents are guaranteed to engage (any dir & pos)
            return True
        if not self.map.g_view.has_edge(u_id, v_id):
            # [TBD] debug for missing target
            return False
        _, prob_raw = self._get_real_prob_by_src_tar(u_id, v_id)
        prob_fin = prob_raw + self.zones[zone_id]["prob_add"]
        # prob_fin = prob_fin * self.zones[zone_id]["prob_mul"]
        # generate a random number in the range of [0., 1.]
        tar_value = uniform(0., 1.)
        # determine if cause damage during this interaction
        return tar_value < prob_fin

    # 2.2.3 get the raw probability for pair-wise engagement
    def _get_real_prob_by_src_tar(self, u_id, v_id):
        u_node = self.engage_mat[u_id, u_id]
        v_node = self.engage_mat[v_id, v_id]
        u_dir = self.engage_mat[u_id, -3]
        u_pos = self.engage_mat[u_id, -2]
        v_pos = self.engage_mat[v_id, -2]
        pos_u_v = self._get_pos_u_v(u_pos, v_pos)
        # get the engagement probability
        edge_num, prob_raw = self.map.get_Gview_prob_by_dir_pos(u_node, v_node, u_dir, pos_u_v)
        return edge_num, prob_raw

    # 2.3 customized observation calculation
    def update_observation(self):
        """ ===> setup custom observation shape & value
        # set values in the teammate and opposite observation slots
        # 1.0 at node
        # 0.5 inside dangerous zone
        # 0.3 inside cautious zone
        # 0.2 in sight
        # 0.1 within machine gun range (team blue only)
        """
        obs_value = [0.1, 0.2, 0.3, 0.5, 1.0]
        death_decay = 0.7
        self.states.reset_step()

        # update elements in team red slot
        for r_id in self.agents.ids_R:
            if self.agents.gid[r_id].death:
                continue
            _dir, _pos, _src = self.engage_mat[r_id, -3:]
            # only check target Standing posture
            _pos_edge = self._get_pos_u_v(_pos, 0)
            neighbors = self.map.get_Gview_neighbor_by_dir_pos(_src, _dir, _pos_edge)
            for _tar in neighbors:
                value = obs_value[self._get_zone_obs(_src, _tar)]
                # set argmax on each node
                if value > self.states.obs_R[_tar - 1]:
                    self.states.obs_R[_tar - 1] = value
            self.states.obs_R[_src - 1] = obs_value[-1]

        # update elements in team blue slot
        node_end = self.configs["field_boundary_node"]  # machine gun coverage area [0 to node_end]
        # add min_val for nodes covered by blue's machine gun squad 
        self.states.obs_B[0:node_end] = obs_value[0]
        for b_id in self.agents.ids_B:
            _dir, _pos, _src = self.engage_mat[b_id, -3:]
            if self.agents.gid[b_id].death:
                self.states.obs_B[_src - 1] = obs_value[-1] * death_decay
                continue
            _pos_edge = self._get_pos_u_v(_pos, 0)
            neighbors = self.map.get_Gview_neighbor_by_dir_pos(_src, _dir, _pos_edge)
            for _tar in neighbors:
                value = obs_value[self._get_zone_obs(_src, _tar)]
                if value > self.states.obs_B[_tar - 1]:
                    self.states.obs_B[_tar - 1] = value
            self.states.obs_B[_src - 1] = obs_value[-1]

        # update states.obs_full using new obs_R and obs_B
        self.states.obs_update()
        return False

    # 2.3.1. get zone token
    def _get_zone_token(self, node_src, node_tar) -> int:
        """ zone token lookup for engage matrix updates
        # check self.configs["engage_token"] for more details
        # if there is an edge in the visibility FOV graph;
        # [!] no self-loop in the visibility graph, check if two agents are on the same node first
        """
        if node_src == node_tar:
            return 4  # overlap
        if self.map.g_view.has_edge(node_src, node_tar):
            dist = self.map.get_Gview_edge_attr_dist(node_src, node_tar)
            # -1 indicates there is no visibility range limitation
            max_range = self.configs["sight_range"]  # or self.zones[1]['dist'] if it is not provided
            if max_range and dist > max_range:
                return 0
            return self._get_zone_by_dist(dist)
        return 0

    # 2.3.2. fast zone token getter for update_observation
    def _get_zone_obs(self, node_src, node_tar) -> int:
        # simple zone token retrieval without checking corner cases
        dist = self.map.get_Gview_edge_attr_dist(node_src, node_tar)
        return self._get_zone_by_dist(dist)

    # 2.3.3. distance based zone token getter
    def _get_zone_by_dist(self, dist) -> int:
        # default_ranges = [dangerous_zone_dist_boundary=50, cautious_zone_dist_boundary=150]
        obs_range = [self.zones[3]["dist"], self.zones[2]["dist"]]
        return 1 if dist > obs_range[1] else (2 if dist > obs_range[0] else 3)

    @staticmethod
    def _get_pos_u_v(pos_u, pos_v):
        # find edge label "pos" by source/target in range {0:"Stand", 1:"Prone"} -> [0,1,2,3]
        return pos_u + pos_u + pos_v

    @staticmethod
    def _get_pos_src(pos_index):
        # find source pos -> {[0,1]:"Stand", [2,3]:"Prone"}
        return int(pos_index < 2)

    # 3.1. step rewards -> list of lists
    def get_step_rewards(self, prev_rewards):
        rewards = prev_rewards
        rew_cfg = self.rew_cfg["step"]
        _col = self.step_counter  # step_num in range [1, max_step]
        for _index, _aid in enumerate(self.agents.ids_ob):
            _agent = self.agents.gid[_aid]
            if _agent.engaged_step:
                rewards[_index] += rew_cfg["rew_step_slow"]
                _dmg_taken = _agent.dmg_step_taken
                _dmg_given = _agent.dmg_step_given
                if _dmg_given:
                    if _dmg_taken < _dmg_given:
                        rewards[_index] += rew_cfg["rew_step_adv"]
                    else:
                        rewards[_index] += rew_cfg["rew_step_dis"]
                self.states.rewards[_index, _col] = rewards[_index]
        return rewards

    # 3.2. episode rewards -> list of lists
    def get_episode_rewards(self):
        rewards = [0] * self.states.num
        rew_cfg = self.rew_cfg["episode"]
        # TBD (lv1): only update red rewards now; should support both teams.
        agent_alive = set()
        agent_death = set()
        health_sum = 0
        health_max = 0
        # gather final states
        for _index, _aid in enumerate(self.agents.ids_ob):
            _agent = self.agents.gid[_aid]
            # skip blue agents
            if _agent.team:
                continue
            _HP = _agent.health
            health_max += _agent.health_max
            if _HP:
                agent_alive.add(_index)
                health_sum += _HP
            else:
                agent_death.add(_index)
        # team based final reward for all agents in team red
        team_delay = self._get_delay_reward_by_step(**rew_cfg["rew_ep_delay"])
        health_index = self._get_val_from_list(health_sum/health_max, rew_cfg["rew_ep_health"]["bar"])
        team_health = rew_cfg["rew_ep_health"]["num"][health_index]
        total_reward = team_delay + team_health
        if len(agent_death):
            # do not award global health reward for death reds
            for _id in agent_death:
                self.states.rewards[_id, 0] = team_delay
                rewards[_id] += team_delay
        elif self.step_counter > rew_cfg["rew_ep_bonus"]["bar"]:
            # bonus reward is conditioned on both health (all alive) and delay (> certain timestep threshold)
            total_reward += rew_cfg["rew_ep_bonus"]["value"]
        for _id in agent_alive:
            self.states.rewards[_id, 0] = total_reward
            rewards[_id] += total_reward
        return rewards

    def _get_delay_reward_by_step(self, **val_dict):
        delay_reward = val_dict["min"]
        _index = self._get_val_from_list(self.step_counter, val_dict["step"])
        if _index:
            for _pre in range(1, _index):
                delay_reward += (val_dict["step"][_pre] - val_dict["step"][_pre - 1]) * val_dict["inc"][_pre]
            delay_reward += (self.step_counter - val_dict["step"][_index - 1]) * val_dict["inc"][_index - 1]
        else:
            delay_reward += (self.step_counter - val_dict["step"][0]) * val_dict["inc"][0]
        if delay_reward > val_dict["max"]:
            delay_reward = val_dict["max"]
        return delay_reward

    @staticmethod
    def _get_val_from_list(target_val, list_val):
        # find the index of the first element in the val_list that is greater than the target value
        return next(_id for _id, _val in enumerate(list_val) if target_val <= _val)

    @staticmethod
    def _get_reward_from_segments(n_step, **val_dict):
        # segment reward range: [max (from step_0 to pivot_step) -> (linear -rew_decay per step) -> 0]
        rew_min = val_dict["min"]
        rew_max = val_dict["max"]
        rew_per_step = val_dict["inc"]
        start_step = val_dict["start_step"]
        if n_step > start_step:
            _reward = rew_min + int((n_step - start_step) * rew_per_step)
            reward = _reward if _reward < rew_max else rew_max
        else:
            reward = rew_min
        return reward

    # 3.3. step done tokens -> list of lists
    def get_done_list(self, force=False):
        # get forced early stop signal
        if force:
            self.states.done_array[:] = True
            return [True] * self.states.num
        else:
            list_done = self.get_step_done()
            self.states.done_array[:] = list_done
            return list_done

    def get_step_done(self):
        # reach to max_step
        if self.step_counter >= self.max_step:
            return [True] * self.states.num
        # all team_Blue or team_Red agents got terminated
        if all([self.agents.gid[_id].death for _id in self.agents.ids_R]) or all([self.agents.gid[_id].death for _id in self.agents.ids_B]):
            return [True] * self.states.num
        # death => done for each observing agent (early termination)
        return [self.agents.gid[_id].death for _id in self.agents.ids_ob]

    # 0.1. load configs and update local defaults
    def _init_env_config(self, **kwargs):
        """ set default env config values if not specified in outer configs """
        from graph_scout.envs.utils.config.default_configs import init_setup as env_cfg
        from copy import deepcopy
        self.configs = deepcopy(env_cfg["INIT_ENV"])
        self.rew_cfg = deepcopy(env_cfg["INIT_REWARD"])
        self.log_cfg = {}

        _config_local = env_cfg["INIT_LOCAL"]
        _config_all = list(_config_local.keys()) + list(self.configs.keys())
        _reward_step = list(self.rew_cfg["step"].keys())
        _reward_epic = list(self.rew_cfg["episode"].keys())
        _log = list(env_cfg["INIT_LOG"].keys())

        # loading outer args and overwrite local default configs
        for key, value in kwargs.items():
            if key in _config_all:
                self.configs[key] = value
            elif key in _reward_step:
                self.rew_cfg["step"][key] = value
            elif key in _reward_epic:
                self.rew_cfg["episode"][key] = value
            elif key in _log:
                self.log_cfg[key] = value
            else:
                # pass
                continue
                raise KeyError("[GSMEnv][Init] Invalid config")

        # set defaults bool vars if not specified
        for key in _config_local:
            if key in self.configs:
                continue
            else:
                self.configs[key] = _config_local[key]
            # load content if True
            if self.configs[key]:
                _new_key = env_cfg["LOCAL_TRANS"][key]
                self.configs[_new_key] = env_cfg["LOCAL_CONTENT"][_new_key]

        # unwrapping eazy access args -> most frequently visited args
        self.max_step = self.configs["max_step"]
        self.invalid_masked = self.configs["masked_act"]

        # setup log inits if not provided
        self.logger = self.configs["log_on"] if "log_on" in self.configs else False
        if self.logger:
            # self.log_cfg["root_path"] = self.configs["env_path"]
            for key in _log:
                if key not in self.log_cfg:
                    self.log_cfg[key] = env_cfg["INIT_LOG"][key]
        return True

    # 0.2. load terrain graphs. [*]executable after calling self._init_env_config()
    def _load_map_graph(self):
        from graph_scout.envs.data.file_manager import load_graph_files
        from graph_scout.envs.data.terrain_graph import MapInfo
        # load graphs
        self.map = MapInfo()
        self.map = load_graph_files(env_path=self.configs["env_path"], map_lookup=self.configs["map_id"])
        # TBD(lv3): call generate_files if parsed data not exist

    # 0.3. initialize all agents. [*] executable after calling self._init_env_config()
    def _init_agent_state(self):
        # init agent instances
        self.agents = AgentManager(self.configs["num_red"], self.configs["num_blue"],
                                   self.configs["health_red"], self.configs["health_blue"],
                                   **self.configs["agents_init"])
        _id, _name, _team = self.agents.get_observing_agent_info()
        # init state instance < default: agents have identical observation shape [teammate slot + opposite slot] >
        self.states = StateManager(len(self.agents.ids_ob), self.map.get_graph_size(), self.max_step,
                                   _id, _name, _team)
        # lookup dicts for all action branches
        self.acts = actsEval()
        self.act_move_set = set(self.acts.move.keys())
        if self.invalid_masked:
            # size of the mask for each agent is the flattened action tensor
            self.action_mask = np.zeros((self.states.num, sum(self.acts.shape())), dtype=bool)

        # generate engagement matrix for all agent-pairs + [dir, pos, target_node_after_move] per agent
        self.engage_mat = np.zeros((self.agents.n_all, self.agents.n_all + 3), dtype=int)
        # init current status elements for all agents
        for _a in range(self.agents.n_all):
            _node, _dir, _pos = self.agents.gid[_a].get_geo_tuple()
            self.engage_mat[_a, _a] = _node
            self.engage_mat[_a, -3:] = [_dir, _pos, _node]
        self.zones = self.configs["engage_range"]

        # memory of heuristic agents [branch, signal_e, signal_h, target_agent, graph_dist, target_node]
        # _dts = len(self.agents.ids_dt)
        # self.buffer_mat = np.zeros((self.configs["buffer_size"], _dts, 5), dtype=int)
        # self.buffer_ptr = [-1] * _dts
        # simple version -> count_down num
        self.dt_buffer = {}
        for a_id in self.agents.ids_dt:
            self.dt_buffer[a_id] = self.configs["buffer_count"]
        # pair-wised assisting matrix + [assist agent_id (>0)]
        n_dt = len(self.agents.ids_dt)
        self.assist_mat = np.zeros((n_dt, n_dt + 1), dtype=int)
        return False

    def _log_step_states(self):
        return self.states.dump_dict() if self.logger else []

    def render(self, mode='human'):
        pass

    # delete local instances
    def close(self):
        if self.agents is not None:
            del self.agents
        if self.states is not None:
            del self.states
        if self.map is not None:
            del self.map


class AgentManager:
    def __init__(self, n_red=0, n_blue=0, health_red=0, health_blue=0, **agent_config):
        # list of all agent object instances (sorted by global_id)
        self.gid = list()

        # numbers of agents
        self.n_all = n_red + n_blue

        # args for resetting agents:
        self.list_init = list()  # [node, 0(motion), dir, posture, health]
        self.dict_path = dict()  # designated paths

        # gid lookup lists
        self.ids_ob = list()  # index of learning agents
        self.ids_dt = list()  # index of heuristic agents
        self.ids_R = list()  # team_id == 0 (red)
        self.ids_B = list()  # team_id == 1 (blue)

        self._load_init_configs(n_red, n_blue, health_red, health_blue, **agent_config)

    def _load_init_configs(self, n_red, n_blue, health_red, health_blue, **agent_config):
        # agent global_id is indexing from 0
        g_id = 0
        default_HP = [health_red, health_blue]
        for _d in agent_config:
            # dict key: agent name (i.e. "red_0", "blue_1")
            _name = _d

            # dict values for each agent: team, type, node/path, acts, HP and tokens
            a_dict = agent_config[_d]
            _type = a_dict["type"]
            _team = a_dict["team_id"]
            _dir = a_dict["direction"]
            _pos = a_dict["posture"]
            _HP = a_dict["health"] if "health" in a_dict else default_HP[_team]

            # learning agents
            if _type == "RL":
                _node = a_dict["node"]
                self.gid.append(AgentCoop(global_id=g_id, name=_name, team_id=_team, health=_HP,
                                          node=_node, direction=_dir, posture=_pos,
                                          _learning=a_dict["is_lr"],
                                          _observing=a_dict["is_ob"]))
                self.ids_ob.append(g_id)
                self.list_init.append([_node, 0, _dir, _pos, _HP])

            # pre-determined agents
            elif _type == "DT":
                _path = a_dict["path"]
                self.gid.append(AgentHeur(global_id=g_id, name=_name, team_id=_team, health=_HP,
                                          path=_path, direction=_dir, posture=_pos))
                self.ids_dt.append(g_id)
                self.list_init.append([0, 0, _dir, _pos, _HP])
                self.dict_path[g_id] = _path

            # TBD(c): default agent & behavior
            else:
                raise ValueError(f"[GSMEnv][Init] Unexpected agent type: {_type}")

            # update team lookup lists
            if _team:
                self.ids_B.append(g_id)
            else:
                self.ids_R.append(g_id)

            g_id += 1

        # TBD(a): add default RL agents if not enough agent_init_configs were provided
        # or just raise error
        #print(n_red, n_blue)
        #print(len(self.ids_R), len(self.ids_B))
        if len(self.ids_R) != n_red or len(self.ids_B) != n_blue:
            raise ValueError("[GSMEnv][Init] Not enough agent init configs were provided.")

    def reset(self):
        for _id in self.ids_ob:
            self.gid[_id].reset(list_states=self.list_init[_id][0:-1],
                                health=self.list_init[_id][-1])
        for _id in self.ids_dt:
            self.gid[_id].reset(list_states=self.list_init[_id][0:-1],
                                health=self.list_init[_id][-1],
                                path=self.dict_path[_id])

    def get_observing_agent_info(self):
        list_name = []
        list_team = []
        for _id in self.ids_ob:
            list_name.append(self.gid[_id].name)
            list_team.append(self.gid[_id].team)
        return self.ids_ob, list_name, list_team

    def close(self):
        del self.gid


class StateManager:
    def __init__(self, num=0, shape=0, max_step=0, ids=None, names=None, teams=None):
        self.num = num
        self.shape = shape
        # agent lookup
        self.a_id = [0] if ids is None else ids
        # dict keys
        self.name_list = [0] if names is None else names
        self.team_list = [0] if teams is None else teams  # {0: "red", 1: "blue"}
        self.team_both = any(self.team_list)
        # dict values
        # observation slots for teammate and opposite [single copy]
        self.obs_R = np.zeros(shape)
        self.obs_B = np.zeros(shape)
        self.obs_full = np.zeros((num, shape))
        self.rewards = np.zeros((num, max_step + 1))
        self.done_array = np.zeros(num, dtype=bool)

    def reset(self):
        self.reset_step()
        self.rewards[:] = 0.
        self.done_array[:] = False

    def reset_step(self):
        self.obs_R[:] = 0.
        self.obs_B[:] = 0.

    def obs_update(self):
        # shared team observation from red's perspective
        obs_team_R = np.concatenate((self.obs_R, self.obs_B), axis=None)
        # fill up observation lists for each observing agent
        if self.team_both:
            # shared team observation from blue's perspective
            obs_team_B = np.concatenate((self.obs_B, self.obs_R), axis=None)
            for index in range(self.num):
                self.obs_full[index] = obs_team_B if self.team_list[index] else obs_team_R
        else:
            # fast copy for all red condition
            self.obs_full = np.tile(obs_team_R, (self.num, 1))

    def get_reward_list(self, step):
        return self.rewards[:, step].tolist()

    def dump_dict(self, step=0):
        _dict_obs = {}
        _dict_rew = {}
        _dict_done = {}
        for _id in range(self.num):
            _key = self.name_list[_id]
            _dict_obs[_key] = self.obs_full[_id]
            _dict_rew[_key] = self.rewards[_id, step]
            _dict_done[_key] = self.done_array[_id]
        return _dict_obs, _dict_rew, _dict_done