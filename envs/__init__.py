from gym.envs.registration import register

#flocking ST
register(
    id='FlockingRelativeST-v0',
    entry_point='envs.flocking:FlockingRelativeSTEnv',
    max_episode_steps=300,
)
register(
    id='FlockingRelative-v0',
    entry_point='envs.flocking:FlockingRelativeEnv',
    max_episode_steps=300,
)

register(
    id='FlockingLeader-v0',
    entry_point='envs.flocking:FlockingLeaderEnv',
    max_episode_steps=1000,
)
# new leader
register(
    id='FlockingLeaderST-v0',
    entry_point='envs.flocking:FlockingLeaderSTNewEnv',
    max_episode_steps=2000,
)