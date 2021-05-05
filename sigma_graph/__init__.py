from gym.envs.registration import register

register(
    id='figure8squad-v0',
    entry_point='sigma_graph.envs.figure8:Figure8Squad',
    max_episode_steps=100,
)

# register(
#     id='figure8mabits-v0',
#     entry_point='sigma_graph.envs.figure8:Figure8Squad6b',
#     max_episode_steps=100,
# )
