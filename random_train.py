from agent import RandomAgent
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

def train(agent, env, actions, optimizer):
  eval_steps = 2000
  max_steps = 100
  eval_reward=[]
  for training_steps in range(max_steps):
          curr_reward = 0.0
          for i in range(eval_steps):            
            action, reward = agent.step()
            curr_reward+=reward
          eval_reward.append(curr_reward)           

    

  cPickle.dump(eval_reward,open('outputs/eval_random_reward.pkl','w'))
  #plot_reward(eval_reward)
    

# cumulative reward for training and test 

def main():
  env = Environment(config2)
  agent = RandomAgent(env)
  
 
  optimizer = None
  train(agent, env, [0,1,2,3], optimizer)

if __name__ == '__main__':
  main()

