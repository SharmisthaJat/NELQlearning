import gym
env = gym.make('SpaceInvaders-v0')
env = gym.wrappers.Monitor(env, '/tmp/test')
env.reset()
env.step(0)
