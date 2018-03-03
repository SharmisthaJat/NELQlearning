import time

import gym

env_name = "SpaceInvadersDeterministic-v3"
env_skip = gym.make(env_name)

x = env_skip.reset()

actions = [2, 2, 2, 5, 5, 5]
ind = 0
while 1:
  env_skip.render()
  action = actions[ind]
  print("doing action ", action)
  env_skip.step(action)
  time.sleep(0.02)
  ind += 1
  if ind >= len(actions):
    ind = 0
