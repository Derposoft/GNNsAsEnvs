# default hyper-parameters for rewards
REWARDS = {
    "reward_done_lookup": [32, 16, 8, 4, 2, 0],
    "reward_R_2_B": 4,
    "reward_B_2_R": -3,
    "reward_R_2_R": -2,
    "reward_fast": 20,
    "step_pivot": 15,
    "step_decay": 1,
}


def get_step_reward(r_engages_b, b_engages_r, **rewards):
    step_reward = 0
    if r_engages_b:
        step_reward += rewards["reward_R_2_B"] if "reward_R_2_B" in rewards.keys() else REWARDS["reward_R_2_B"]
    if b_engages_r:
        step_reward += rewards["reward_B_2_R"] if "reward_B_2_R" in rewards.keys() else REWARDS["reward_B_2_R"]
    return step_reward


def get_step_overlay(overlay, **rewards):
    overlay_reward = 0
    if overlay:
        overlay_reward += rewards["reward_R_2_R"] if "reward_R_2_R" in rewards.keys() else REWARDS["reward_R_2_Rs"]
    return overlay_reward


def get_episode_reward_agent(health_self, health_blue, init_health, damage_thres_r=5, damage_thres_b=5, agent_damage=0,
                             step_end=-1, **args):
    episode_reward = 0
    done_reward = args["reward_done_lookup"] if "reward_done_lookup" in args.keys() else REWARDS["reward_done_lookup"]
    # reward for surviving while terminating the blue agent
    if health_blue <= (init_health - damage_thres_b) and (init_health - damage_thres_r) < health_self:
        episode_reward += done_reward[init_health - health_self]
        # speed reward for early termination [segment function X(step:0) -> X(step:pivot) ->(linear decay) 0]
        if step_end > 0:
        # if step_end > 0 and agent_damage >= damage_thres_r:
            _reward_fast = args["reward_fast"] if "reward_fast" in args.keys() else REWARDS["reward_fast"]
            _step_pivot = args["step_pivot"] if "step_pivot" in args.keys() else REWARDS["step_pivot"]
            _step_decay = args["step_decay"] if "step_decay" in args.keys() else REWARDS["step_decay"]
            if step_end > _step_pivot:
                _reward = _reward_fast - (step_end - _step_pivot) * _step_decay
                episode_reward += _reward if _reward > 0 else 0
            else:
                episode_reward += _reward_fast
    return episode_reward


def get_episode_reward_team(health_R, health_B, damage_R, endtime_B, health_init,
                            damage_thres_team_R=5, damage_thres_team_B=5, **args):
    episode_reward = 0
    # TODO>> team based rewards
    return episode_reward
