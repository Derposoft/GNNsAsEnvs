""" global arguments: lookup table for movements and directions """


class ActionBranched:
    def __init__(self, is_4_dirs=True):
        # lookup dicts
        self.move = MOVE_LOOKUP
        self.look = TURN_4_LOOKUP if is_4_dirs else TURN_3_LOOKUP
        self.body = POSTURE_LOOKUP
        # range [0, n)
        self.n_move = len(self.move)
        self.n_look = len(self.look)
        self.n_body = len(self.body)

    def shape(self):
        return [self.n_move, self.n_look, self.n_body]


# move actions
MOVE_LOOKUP = {
    0: "NOOP",
    1: "N",
    2: "S",
    3: "W",
    4: "E",
}

# [default] turn actions are not correlated with move actions
TURN_4_LOOKUP = {
    0: 1,   # "N"
    1: 2,   # "S"
    2: 3,   # "W"
    3: 4,   # "E"
}

# turn actions are conditioned on the current looking dir (after moving)
TURN_3_LOOKUP = {
    0: {1: 1, 2: 2, 3: 3, 4: 4},  # "NOOP",
    1: {1: 3, 2: 4, 3: 2, 4: 1},  # "TURN_LEFT_90" dir after turning left for 90 degree,
    2: {1: 4, 2: 3, 3: 1, 4: 2},  # "TURN_RIGHT_90" dir after turning right for 90 degree,
}

# posture
POSTURE_LOOKUP = {
    0: "Stand",
    1: "Prone",
}