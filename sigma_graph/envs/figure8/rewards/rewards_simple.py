# default hyper-parameters for rewards
REWARDS = {
    "reward_R_2_B": 4,
    "reward_B_2_R": -3,
    "reward_R_2_R": -2,
    "reward_episode": {"reward": [32, 16, 8, 4, 2, 0], "damage": [0, 1, 2, 3, 4, 5]},
    "reward_faster": {"type": "segment", "pivot_step": 10, "reward_init": 16, "reward_decay": 1}
}


def get_step_engage(r_engages_b, b_engages_r, team_switch=False, **rewards):
    step_reward = 0
    if r_engages_b:
        step_reward += rewards["reward_step_RB"] if "reward_step_RB" in rewards else REWARDS["reward_R_2_B"]
    if b_engages_r:
        step_reward += rewards["reward_step_BR"] if "reward_step_BR" in rewards else REWARDS["reward_B_2_R"]
    if team_switch is True:
        step_reward = -step_reward
    return step_reward


def get_step_overlay(overlay, **rewards):
    overlay_reward = 0
    if overlay:
        overlay_reward += rewards["reward_step_RR"] if "reward_step_RR" in rewards else REWARDS["reward_R_2_R"]
    return overlay_reward


def get_episode_reward_agent(full_health, health_self, health_opponent, thres_self, thres_opponent,
                             agent_damage=0, step_end=-1, **rewards):
    episode_reward = 0
    # discourage free loaders
    if agent_damage == 0:   # agent_damage < damage_thres_r:
        return episode_reward

    done_reward = rewards["reward_done_lookup"] if "reward_done_lookup" in rewards else REWARDS["reward_episode"]
    # reward for surviving while terminating the blue agent
    if health_opponent <= (full_health - thres_opponent) and (full_health - thres_self) < health_self:
        damage = full_health - health_self
        if damage >= done_reward["damage"][-1]:
            index = -1
        else:
            # find the index of the fist element in the list that no less than the target damage
            index = next(_idx for _idx, _val in enumerate(done_reward["damage"]) if _val >= damage)
        episode_reward += done_reward["reward"][index]

        # speed reward for early termination
        if step_end > 0:
            _fast = rewards["reward_faster_lookup"] if "reward_faster_lookup" in rewards else REWARDS["reward_faster"]
            if _fast["type"] == "segment":
                # segment reward function: [rew_start (from step 0 to pivot_step) -> (linear rew_decay per step) -> 0]
                _step_pivot = _fast["pivot_step"]
                _reward_start = _fast["reward_init"]
                _reward_decay = _fast["reward_decay"]
                if step_end > _step_pivot:
                    _reward = _reward_start - (step_end - _step_pivot) * _reward_decay
                    episode_reward += _reward if _reward > 0 else 0
                else:
                    episode_reward += _reward_start
            # TODO>>> define other function types
            else:
                assert "Reward function not implemented:{}".format(_fast)
    return episode_reward


def get_episode_reward_team(health_list_r, health_list_b, health_init,
                            damage_thres_r, damage_thres_b, damage_list_r,endtime_list_b, **rewards):
    episode_reward = 0
    # TODO>> team based reward for all agents in the team
    return episode_reward
