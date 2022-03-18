class MAgent:
    def __init__(self, _uid=0, _node=0, _code=None, _dir=0, _health=0, _learn=True):
        self.agent_id = _uid
        self.agent_node = _node
        self.agent_code = _code
        self.agent_dir = _dir
        self.health = _health
        self.learnable = _learn
        self.frozen = False

    def set_location(self, _node, _code, _dir):
        self.agent_node = _node
        self.agent_code = _code
        self.agent_dir = _dir

    def get_pos_dir(self):
        return [self.agent_node, self.agent_dir]

    def get_encoding(self):
        return self.agent_code

    def get_id(self):
        return self.agent_id

    def get_health(self):
        return self.health

    def take_damage(self, points):
        self.health -= points
        if self.health <= 0:
            self.frozen = True

    def is_learning(self):
        return self.learnable

    def is_frozen(self):
        return self.frozen

    def set_frozen(self, _binary_flag):
        self.frozen = _binary_flag

    def reset(self, _node, _code, _dir, _health):
        self.agent_node = _node
        self.agent_code = _code
        self.agent_dir = _dir
        self.health = _health
        self.frozen = False

    def close(self):
        pass
