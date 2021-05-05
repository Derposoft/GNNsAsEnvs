class MAgent:
    def __init__(self, _uid=0, _node=0, _code=None, _dir=0, _health=5):
        self.agent_id = _uid
        self.agent_node = _node
        self.agent_code = _code
        self.agent_dir = _dir
        self.health = _health

    def set_location(self, _node, _code, _dir):
        self.agent_node = _node
        self.agent_code = _code
        self.agent_dir = _dir

    def get_pos_dir(self):
        return [self.agent_node, self.agent_dir]

    def get_encoding(self):
        return self.agent_code

    def get_health(self):
        return self.health

    def take_damage(self, points):
        self.health -= points
        return self.health

    def reset(self, _node, _code, _dir, _health):
        self.agent_node = _node
        self.agent_code = _code
        self.agent_dir = _dir
        self.health = _health

    def close(self):
        pass
