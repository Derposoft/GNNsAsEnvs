""" global arguments: lookup table for movements and directions """

# move actions
MOVE_LOOKUP = {
    0: "NOOP",
    1: "N",
    2: "S",
    3: "W",
    4: "E",
}

# turn actions conditioned on +/- 90 degree to the move action
TURN_90_LOOKUP = {
    0: "NOOP",
    1: "TURN_LEFT_90",
    2: "TURN_RIGHT_90",
}
TURN_L = {1: 3, 2: 4, 3: 2, 4: 1}  # looking dir after turning left for 90 degree
TURN_R = {1: 4, 2: 3, 3: 1, 4: 2}  # looking dir after turning right for 90 degree

# turn actions not correlated with move actions
TURN_4_LOOKUP = {
    0: 1,   # "N"
    1: 2,   # "S"
    2: 3,   # "W"
    3: 4,   # "E"
}
