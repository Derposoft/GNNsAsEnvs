# lookup table for movements and directions
ACTION_LOOKUP = {
    0: "NOOP",
    1: "N",
    2: "S",
    3: "W",
    4: "E"
}

ACTION_TURN_LOOKUP = {
    0: "NOOP",
    1: "TURN_LEFT_90",
    2: "TURN_RIGHT_90",
}
TURN_L = {1: 3, 2: 4, 3: 2, 4: 1}
TURN_R = {1: 4, 2: 3, 3: 1, 4: 2}

# interaction behavior settings
INTERACT_LOOKUP = {
    "sight_range": -1,  # -1 for unlimited range
    "engage_range": 25,
    "engage_damage": 1,
    "damage_threshold": 5,
    "unmasked_invalid_action_penalty": -10
}

# select default spawn positions for red agents
INIT_POS_LOOKUP = [
    [(11, 0), (11, 1), (11, 2), (12, 3), (13, 3), (14, 3), (14, 4)],     # 7 nodes on the right side of the map
    [(14, 7), (14, 8), (13, 5), (12, 5), (11, 4), (11, 5), (11, 6)]      # 7 nodes on the left side of the map
]
# "S": [(11, 0), (11, 1), (11, 2), (12, 3), (12, 4), (13, 3), (13, 4), (14, 3), (14, 4), (14, 5)] # init pos for red

# select which map to use
MAP_LOOKUP = {
    "S": "_27",
    "M": "_41",
    "L": "_73",
}
