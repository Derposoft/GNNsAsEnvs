from . import default_setup as env_setup
from . import figure8_squad as base_env

local_action_move = env_setup.act.MOVE_LOOKUP
local_action_turn = env_setup.act.TURN_4_LOOKUP


class Figure8Squad4Dir(base_env.Figure8Squad):
    def __init__(self, _max_step=40, _n_red=2, _n_blue=1, **kwargs):
        super().__init__(max_step=_max_step, n_red=_n_red, n_blue=_n_blue, **kwargs)

    def _take_action_red(self, n_actions):
        action_penalty = [0] * self.num_red
        _action_stay_penalty = self.configs["penalty_stay"] if "penalty_stay" in self.configs else 0

        for agent_i, actions in enumerate(n_actions):
            # check input action if in range of the desired discrete action space
            assert self.action_space[agent_i].contains(actions), f"{actions}: action out of range"
            if self.team_red[agent_i].is_frozen():
                continue
            action_move, action_turn = actions
            # find all 1st ordered neighbors of the current node
            agent_encode = self.team_red[agent_i].agent_code
            prev_idx, list_neighbor, list_act = self.map.get_all_states_by_node(agent_encode)
            # validate actions with masks
            if action_move != 0 and action_move not in list_act:
                # if the learning process doesn't have masks, then the invalid Action_Move should be replaced by 'NOOP'.
                if not self.invalid_masked:
                    action_move = 0
                    action_penalty[agent_i] = self.configs["penalty_invalid"]
                # if the masking turned on in the learning, invalid actions should not appear.
                else:
                    assert f"[ActError] act{action_move} pos{prev_idx} masking{self.action_mask[agent_i]}"

            # make 'Move' and 'Turn' actions
            if action_move == 0:
                if _action_stay_penalty:
                    action_penalty[agent_i] += _action_stay_penalty
            elif action_move in list_act:
                _node = list_neighbor[list_act.index(action_move)]
                self.team_red[agent_i].agent_node = _node
                self.team_red[agent_i].agent_code = self.map.get_name_by_index(_node)
            self.team_red[agent_i].agent_dir = local_action_turn[action_turn]
        return action_penalty

    # def _take_action_blue(self):
    #     pass

    # def _update(self):
    #     pass

    # def _step_rewards(self):
    #     pass

    # def _episode_rewards(self):
    #     pass

    # def _get_step_done(self):
    #     pass
