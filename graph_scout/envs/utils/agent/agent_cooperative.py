from graph_scout.envs.utils.agent.multiagent_base import GSMAgent


class AgentCoop(GSMAgent):
    def __init__(self, global_id=0, name="R0", team_id=0, node=0,
                 motion=0, direction=0, posture=0, health=100, _death=False,
                 _learning=True, _observing=True):
        super().__init__(global_id, name, team_id, node,
                         motion, direction, posture, health, _death)
        # interactive args
        self.engaged_total = 0
        # step args for mini-steps sum
        self.dmg_step_taken = 0
        self.dmg_step_given = 0
        self.engaged_step = 0
        # RL control args
        self.is_learning = _learning
        self.is_observing = _observing

    # A binary token for active learning or frozen status
    @property
    def is_learning(self):
        return self._learning

    @is_learning.setter
    def is_learning(self, value):
        if value != bool(value):
            raise TypeError("[GSMEnv][Agent] value must be a bool")
        self._learning = value

    # A binary token for hidden/blind agents
    @property
    def is_observing(self):
        return self._observing

    @is_observing.setter
    def is_observing(self, value):
        if value != bool(value):
            raise TypeError("[GSMEnv][Agent] value must be a bool")
        self._observing = value

    def reset(self, list_states, health=100, _death=False, _learning=True, _observing=True):
        super().reset(list_states, health, _death)
        self.engaged_total = 0
        self.is_learning = _learning
        self.is_observing = _observing
        self.step_reset()

    def step_reset(self):
        self.dmg_step_taken = 0
        self.dmg_step_given = 0
        self.engaged_step = 0

    # major engagements only
    def damage_taken(self, num_point):
        super().damage_taken(num_point)
        self.dmg_step_taken += num_point

    def damage_given(self, num_point):
        super().damage_given(num_point)
        self.dmg_step_given += num_point

    # minor interactions (missing shots)
    def disturbing(self, num_count=1):
        self.engaged_total += num_count
        self.engaged_step += num_count