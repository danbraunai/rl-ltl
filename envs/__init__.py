from gym.envs.registration import register

register(
    id='FrozenLake-v1',
    entry_point='envs.frozen_lake:FrozenLake',
    kwargs={'map_name' : '5x5'},
    max_episode_steps=100
)