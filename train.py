from agent import RLAgent
from environment import Environment
from config import config2
from plot import plot_reward
import nel

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

def train(agent, env, actions, optimizer):
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
  eval_steps = 2000
  max_steps = 100001
  for training_steps in range(max_steps):
    if training_steps < EPS_DECAY_START:
      epsilon = EPS_START
    elif training_steps > EPS_DECAY_END:
      epsilon = EPS_END
    else:
      epsilon = EPS_START - (EPS_START - EPS_END) * (training_steps - EPS_DECAY_START) / (EPS_DECAY_END - EPS_DECAY_START)

    add_to_replay = len(agent.prev_states) == 3
    s1 = agent.get_state()
    action, reward = agent.step(epsilon)
    s2 = agent.get_state()
    if add_to_replay:
      replay.append((s1, action.value, reward, s2))  # enum issue in server machine
      #replay.append((s1, action, reward, s2))

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
        if training_steps % (update_frequency * 100) == 0:
          print(loss.data[0])

    if training_steps % target_update_frequency == 0:
      agent.target.load_state_dict(agent.policy.state_dict())

    if training_steps % eval_frequency == 0:
      env_eval = Environment(config2)
      agent_eval = RLAgent(env_eval)          
      #painter = nel.MapVisualizer(env_eval.simulator, config2, (-30, -30), (150, 150))
      agent_eval.policy.load_state_dict(agent.policy.state_dict())
      curr_reward = 0.0
      for i in range(eval_steps):            
        s1 = agent_eval.get_state()
        action, reward = agent_eval.step()
        curr_reward+=reward
        #painter.draw()
      eval_reward.append(curr_reward)           

    
  cPickle.dump(eval_reward,open('outputs/eval_reward.pkl','w'))
  plot_reward(eval_reward,'RL_agent_eval')
  print(eval_reward[-1])

  env_eval = Environment(config2)
  agent_eval = RLAgent(env_eval)          
  painter = nel.MapVisualizer(env_eval.simulator, config2, (-30, -30), (150, 150))
  agent_eval.policy.load_state_dict(agent.policy.state_dict())
  cur_reward = 0
  for i in range(100):            
    s1 = agent_eval.get_state()
    action, reward = agent_eval.step()
    print(reward)
    curr_reward+=reward
    painter.draw()
  print(cur_reward)

    

# cumulative reward for training and test 

def main():
  env = Environment(config2)
  agent = RLAgent(env)
  
 
  optimizer = optim.SGD(agent.policy.parameters(), lr=.1)
  train(agent, env, [0,1,2,3], optimizer)

if __name__ == '__main__':
  main()

