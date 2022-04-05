from random import randint

from . import action_lookup as act
from .maps.map_configs import INIT_POS_LOOKUP, INIT_REGION
from sigma_graph.data.data_helper import get_node_name_from_pos_abs, get_pos_norms, check_pos_abs_range
from .rewards.rewards_simple import DEFAULT_REWARDS

ACT_LOOK_DIR = [key for key, value in act.MOVE_LOOKUP.items() if value != "NOOP"]
ACT_MASKED = {
    "mask_on": True,
    "unmasked_invalid_action_penalty": -10,
}

INIT_AGENT_RED = {"learn": True, "pos": None, "dir": None}
INIT_AGENT_BLUE = {"learn": False, "route": "0", "idx": 0}
INIT_CONFIGS = {
    "env_path": './', "map_id": 'S', "load_pickle": True,
    "init_red": None, "init_blue": None, "init_health_red": 10, "init_health_blue": 100, "damage_maximum": 5,
    "fixed_start": -1
}
INIT_CONFIGS_LOCAL = ["threshold_damage_2_red", "threshold_damage_2_blue", "act_masked", "penalty_stay"]

# Designed for eval logger. default path is ./logs/ no log files should be generated during training steps.
INIT_LOGS = {
    "log_on": False, "log_path": "logs/", "log_prefix": "log_", "log_overview": "reward_episodes.txt",
    "log_verbose": False, "log_plot": False, "log_save": True,
}

INIT_REWARDS = DEFAULT_REWARDS

# interaction settings
INTERACT_LOOKUP = {
    "sight_range": -1,  # -1 for unlimited range
    "engage_range": 25,
    "engage_behavior": {"damage": 1, "probability": 1.0},
}

# tokens for observation spaces generation
OBS_TOKEN = {
    "obs_embed": False,  # _is_pos_encoded
    "obs_dir": False,  # _has_self_dir
    "obs_sight": False,  # _has_obs_sight_check
    "obs_range": False,  # _has_obs_range_check
    "obs_team": True,  # _has_obs_teammate_pos
    "obs_graph": True, # _is_graph_ready
}


# update agent position. return value: str of binary embedding code
def get_default_red_encoding(red_id=0, red_pos=None) -> str:
    # providing the coordinates of red position (row, col) or sampling from init spawn position pools
    if red_pos is None:
        _region = red_id
    elif isinstance(red_pos, int):
        _region = red_pos
    elif isinstance(red_pos, tuple):
        return get_node_name_from_pos_abs(red_pos)
    else:
        assert "Unexpected POS flags"
    pos_red_pool = INIT_POS_LOOKUP[_region % len(INIT_POS_LOOKUP)]
    idx = randint(0, len(pos_red_pool) - 1)
    R_pos = pos_red_pool[idx]
    return get_node_name_from_pos_abs(R_pos)


# update agent looking direction. return value: int direction indicator
def get_default_dir(default_dir=None) -> int:
    if default_dir is not None:
        return default_dir
    return randint(min(ACT_LOOK_DIR), max(ACT_LOOK_DIR))


def check_agent_init(team, nums, configs) -> list:
    agent_init_configs = []
    if team == "red":
        if configs is None:
            agent_init_configs = [INIT_AGENT_RED for _ in range(nums)]
        else:
            assert len(configs) == nums, "Invalid config: 'init_red'"
            for idx, _config in enumerate(configs):
                # set up default looking direction "dir":[int]
                if "dir" not in _config:
                    _config["dir"] = None
                # parse "pos" indicator and validate range
                _pos = _config["pos"]
                if _pos is None:
                    continue
                if isinstance(_pos, tuple):
                    assert check_pos_abs_range(_pos), "Pos tuple: {} out of range".format(_pos)
                elif isinstance(_pos, str):
                    assert _pos in INIT_REGION, "Invalid region:\'{}\' (not in {})".format(_pos, INIT_REGION.keys())
                    _config["pos"] = INIT_REGION[_pos]
                agent_init_configs.append(_config)
    elif team == "blue":
        if configs is None:
            agent_init_configs = [INIT_AGENT_BLUE for _ in range(nums)]
        else:
            assert len(configs) == nums, "Invalid config: 'init_blue'"
            for idx, _config in enumerate(configs):
                assert "route" in _config, "Invalid init_blue: must specify a route"
                if "idx" not in _config:
                    _config["idx"] = 0
                agent_init_configs.append(_config)
    else:
        assert "Not implemented."
    return agent_init_configs


# get default observation shape per agent
def get_state_shape(n_G, n_R, n_B, shape_tokens) -> int:
    return sum(get_state_shapes(n_G, n_R, n_B, shape_tokens))

# get default observation shape per agent
def get_state_shapes(n_G, n_R, n_B, shape_tokens) -> int:
    _is_embed, _has_self_dir, _has_sight, _has_range, _has_obs_team, _has_obs_graph = list(shape_tokens.values())
    (bit_row, bit_col), _ = get_pos_norms()
    bit_embed = bit_row + bit_col
    look_dir = len(ACT_LOOK_DIR)
    # agent states for local obs and global view of opponents and teammates
    state_self = (bit_embed if _is_embed else n_G) + _has_self_dir * look_dir + (_has_sight + _has_range) * n_B
    state_B = (((bit_embed + look_dir) * n_B) if _is_embed else (n_G + look_dir * n_B)) + (_has_sight + _has_range) * n_B
    state_R = _has_obs_team * (bit_embed * (n_R - 1) if _is_embed else n_G)
    return state_self, state_B, state_R


