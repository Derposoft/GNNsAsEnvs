class GSMAgent:
    def __init__(self, global_id=0, name="A0", team_id=0, node=0,
                 motion=0, direction=0, posture=0, health=100, _death=False):
        # basic info
        self.id = global_id
        self.name = name
        self.team = team_id
        # map info
        self.at_node = node
        # action states
        self.motion = motion
        self.direction = direction
        self.posture = posture
        # interactive args
        self.damage = 0
        self.health = health
        self.health_max = health
        self.death = _death

    @property
    def death(self):
        return self._death

    @death.setter
    def death(self, value):
        if value != bool(value):
            raise TypeError("[GSMEnv][Agent] value must be a bool")
        self._death = value
    
    # fast updating without verifying values
    def set_states(self, num_list):
        self.at_node = num_list[0]
        self.motion = num_list[1]
        self.direction = num_list[2]
        self.posture = num_list[3]

    def set_acts(self, _move, _dir, _pos):
        self.motion = _move
        self.direction = _dir
        self.posture = _pos

    def get_act_tuple(self):
        return [self.motion, self.direction, self.posture]

    def get_geo_tuple(self):
        return [self.at_node, self.direction, self.posture]

    # health value is greater or equal to 0
    def damage_taken(self, num_deduction):
        if num_deduction < self.health:
            self.health -= num_deduction
        else:
            self.health = 0
            self.death = True

    def damage_given(self, num_point):
        self.damage += num_point

    def reset(self, list_states, health=0, _death=False):
        self.set_states(list_states)
        self.damage = 0
        self.health = health if health else self.health_max
        self.death = _death