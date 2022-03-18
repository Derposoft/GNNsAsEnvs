import gym


# a list of [action spaces]. support Discrete and MultiDiscrete
class ActionSpaces(list):
    def __init__(self, n_action_space):
        for item in n_action_space:
            assert isinstance(item, (gym.spaces.Discrete, gym.spaces.MultiDiscrete))
        super().__init__(n_action_space)
        self._action_space = n_action_space

    def length(self):
        return self._action_space.shape[0]


# a list of [observation spaces]
class ObservationSpaces(list):
    def __init__(self, n_obs_space):
        for item in n_obs_space:
            assert isinstance(item, gym.spaces.Box)
        super().__init__(n_obs_space)
        self._n = len(n_obs_space)
        self._obs_space = n_obs_space

    def shape(self):
        return [self._obs_space[_].shape for _ in range(self._n)]
    
    def sample(self):
        return [obs_space.sample() for obs_space in self._obs_space]

    def contains(self, obs):
        # modified contains in case of rllib multi-agent setup
        if type(obs) == dict:
            _obs = []
            for key in obs:
                _obs.append(obs[key])
            obs = _obs
        for space, ob in zip(self._obs_space, obs):
            if not space.contains(ob):
                return False
        return True
