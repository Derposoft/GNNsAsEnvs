from gym.envs.registration import register

register(
    id='figure8squad-v3',
    entry_point='sigma_graph.envs.figure8:Figure8Squad',
    max_episode_steps=100,
)

register(
    id='figure8squad-v4',
    entry_point='sigma_graph.envs.figure8:Figure8Squad4Dir',
    max_episode_steps=100,
)

register(
    id='figure8squadrllib-v1',
    entry_point='sigma_graph.envs.figure8:Figure8SquadRLLib',
    max_episode_steps=100,
)
