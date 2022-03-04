from math import ceil

# default hyper-parameters for rewards
DEFAULT_REWARDS = {
    "step": {"reward_step_on": True, "red_2_blue": 4, "blue_2_red": -3, "red_overlay": -2, }, # TODO
    #"step": {"reward_step_on": True, "red_2_blue": 0, "blue_2_red": 0, "red_overlay": 0, },
    "episode": {
        "reward_episode_on": True, "episode_decay_soft": True,
        "health_lookup": {"type": "table", "reward": [32, 16, 8, 4, 2, 0], "damage": [0, 1, 2, 3, 4, 100]},
        "faster_lookup": {"type": "segment", "pivot_step": 10, "reward_init": 16, "reward_decay": 1},
        "soft_bound": {"dist": [1, 2], "decay_factor": [0.25, 0.125]}
    },
}


def get_step_engage(r_engages_b, b_engages_r, team_switch=False, **rewards):
    assert len(rewards), "No step rewards provided.."
    step_reward = 0
    if r_engages_b:
        step_reward += rewards["red_2_blue"]
    if b_engages_r:
        step_reward += rewards["blue_2_red"]
    if team_switch is True:
        step_reward = -step_reward
    return step_reward


def get_step_overlay(overlay, **rewards):
    assert len(rewards), "No step rewards provided.."
    return rewards["red_overlay"] if overlay else 0


def get_episode_reward_agent(health_lost_self, health_lost_opponent, threshold_self, threshold_opponent,
                             damage_cost_self=0, end_step_opponent=-1, **rewards):
    assert len(rewards), "No episode rewards provided.."
    episode_reward = 0
    # discourage free loaders
    # if damage_cost_self == 0:
    #     return episode_reward

    threshold_offset = rewards["soft_bound"]["dist"][-1] if rewards["episode_decay_soft"] is True else 0
    # give rewards for terminating the opponent agent
    if health_lost_opponent >= threshold_opponent - threshold_offset:
        # health based reward for surviving
        episode_reward += get_reward_type(health_lost_self, **rewards["health_lookup"])

        # speed based reward for fast termination
        if end_step_opponent > 0:
            episode_reward += get_reward_type(end_step_opponent, **rewards["faster_lookup"])

        # apply soft boundary factor
        if rewards["episode_decay_soft"] is True and health_lost_opponent < threshold_opponent:
            _dist = threshold_opponent - health_lost_opponent
            index = next(_idx for _idx, _val in enumerate(rewards["soft_bound"]["dist"]) if _val >= _dist)
            episode_reward = ceil(episode_reward * rewards["soft_bound"]["decay_factor"][index])

    return episode_reward


def get_reward_type(value, **_dict):
    _reward = 0
    _type = _dict["type"]
    if _type == "none":
        return _reward
    elif _type == "table":
        _reward = get_table_reward(value, **_dict)
    elif _type == "segment":
        _reward = get_segment_reward(value, **_dict)
    else:
        assert f"Reward function <episode:faster> not implemented:{_dict}"
    return _reward


def get_table_reward(damage_taken, **_dict):
    # find the index of the fist element in the list that no less than the target damage
    index = next(_idx for _idx, _val in enumerate(_dict["damage"]) if _val >= damage_taken)
    return _dict["reward"][index]


def get_segment_reward(step, **_dict):
    # segment reward function: [rew_start (from step 0 to pivot_step) -> (linear rew_decay per step) -> 0]
    step_start = _dict["pivot_step"]
    reward_start = _dict["reward_init"]
    reward_decay = _dict["reward_decay"]
    if step > step_start:
        _reward = reward_start - int((step - step_start) * reward_decay)
        reward = _reward if _reward > 0 else 0
    else:
        reward = reward_start
    return reward


def get_episode_reward_team(health_list_r, health_list_b, health_init,
                            damage_thres_r, damage_thres_b, damage_list_r, endtime_list_b, **rewards):
    assert len(rewards), "No episode rewards provided.."
    episode_reward = 0
    # TODO>> team based reward for all agents in the team
    return episode_reward
