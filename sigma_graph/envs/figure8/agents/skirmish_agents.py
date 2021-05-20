from sigma_graph.envs.utils.multiagent_agent_base import MAgent


class AgentRed(MAgent):
    def __init__(self, _uid=0, _node=0, _code=None, _dir=0, _health=0, _learn=True):
        super().__init__(_uid, _node, _code, _dir, _health, _learn)
        self.total_damage = 0

    def reset(self, _node, _code, _dir, _health):
        super().reset(_node=_node, _code=_code, _dir=_dir, _health=_health)
        self.total_damage = 0

    def damage_total(self):
        return self.total_damage

    def damage_add(self, points=1):
        self.total_damage += points

    def close(self):
        pass


class AgentBlue(MAgent):
    def __init__(self, _uid=0, _node=0, _code=None, _dir=0, _health=0, _learn=False, _route=None, _index=None):
        super().__init__(_uid, _node, _code, _dir, _health, _learn)
        self.route_ptr = _route
        self.route_idx = _index
        self.end_step = -1

    def reset(self, _node, _code, _dir, _health, _index=0, _end=-1):
        super().reset(_node=_node, _code=_code, _dir=_dir, _health=_health)
        self.route_idx = _index
        self.end_step = _end

    def get_route(self):
        return self.route_ptr

    def get_index(self):
        return self.route_idx

    def set_end_step(self, step):
        self.end_step = step

    def get_end_step(self):
        return self.end_step

    def update_index(self, _idx, _node, _code, _dir):
        self.route_idx = _idx
        super().set_location(_node, _code, _dir)

    def close(self):
        pass
