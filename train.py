from agent import RLAgent
from environment import Environment
from config import config2
from plot import plot_reward

from collections import deque
import random
import cPickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import numpy as np

def train(agent, env, actions, optimizer, agent_eval, env_eval):
  EPS_START = 1.
  EPS_END = .1
  EPS_DECAY_START=1000.
  EPS_DECAY_END=50000.

  update_frequency = 4
  target_update_frequency = 1000
  eval_frequency = 1000
  batch_size = 16
  training_steps = 0
  epsilon = 1.
  replay = deque(maxlen=10000)
  discount_factor = .9
  eval_reward = []
  eval_steps = 10000
  max_epoch = 100000000
  for i in range(max_epoch):
    if training_steps < EPS_DECAY_START:
      epsilon = EPS_START
    elif training_steps > EPS_DECAY_END:
      epsilon = EPS_END
    else:
      epsilon = EPS_START - (EPS_START - EPS_END) * (training_steps - EPS_DECAY_START) / (EPS_DECAY_END - EPS_DECAY_START)

    add_to_replay = len(agent.prev_states) == 3
    s1 = agent.get_state()
    action, reward = env.step(agent)
    s2 = agent.get_state()
    if add_to_replay:
      #replay.append((s1, action.value, reward, s2))  # enum issue in server machine
      replay.append((s1, action, reward, s2))

    if training_steps % update_frequency == 0:
      if batch_size < len(replay):
        sample = random.sample(replay, batch_size)
        s1, action, reward, s2 = zip(*sample)
        s1 = np.array(s1)
        reward = np.array(reward)
        s2 = np.array(s2)
        action = np.array(action)
        s1 = Variable(torch.from_numpy(s1).float())
        q1 = agent.policy(s1)
        q1 = q1[torch.arange(0, action.size).long(), torch.LongTensor(action)]

        s2 = Variable(torch.from_numpy(s2).float())
        q2 = agent.target(s2).data
        q2, _ = torch.max(q2, 1)

        reward = torch.FloatTensor(reward)
        y = Variable(reward + (discount_factor * q2))

        #huber = nn.SmoothL1Loss()
        mse = nn.MSELoss()
        loss = mse(q1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if training_steps % target_update_frequency == 0:
          agent.target.load_state_dict(agent.policy.state_dict())
          print(loss.data[0])

        if training_steps % eval_frequency == 0:
          agent_eval.policy.load_state_dict(agent.policy.state_dict())
          curr_reward = 0.0
          for i in range(eval_steps):            
            s1 = agent_eval.get_state()
            action, reward = env_eval.step(agent_eval)
            curr_reward+=reward
          eval_reward.append(curr_reward)           

    training_steps += 1

  cPickle.dump(eval_reward,open('outputs/eval_reward.pkl','w'))
  plot_reward(eval_reward)
    

def main():
  env = Environment(config2)
  agent = RLAgent(env)
  env_eval = Environment(config2)
  agent_eval = RLAgent(env_eval)
  optimizer = optim.SGD(agent.policy.parameters(), lr=.1)
  train(agent, env, [0,1,2,3], optimizer, agent_eval, env_eval)

if __name__ == '__main__':
  main()

