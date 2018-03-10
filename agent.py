import numpy as np
from collections import deque

import environment as env
import nel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from IPython import embed

actions = [nel.Direction.UP, nel.Direction.DOWN, nel.Direction.LEFT, nel.Direction.RIGHT]

class Policy(nn.Module):
  def __init__(self, action_dim=len(actions), state_size=30, history=4, hidden_size=64):
    super(Policy, self).__init__()
    self.fc1 = nn.Linear(state_size * history, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, action_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)
    
    

class RLAgent(nel.Agent):
  def __init__(self, env, history=3, load_filepath=None):
    super(RLAgent, self).__init__(env.simulator, load_filepath)
    self.env = env
    self.policy = Policy()
    self.prev_states = deque(maxlen=history + 1)
    self.history = history

  def next_move(self):
    state = self.get_state()
    self.prev_states.append(state)
    if(len(self.prev_states) < self.history + 1):
      return actions[np.random.randint(0, len(actions))]

    context = Variable(torch.from_numpy(np.concatenate(self.prev_states)))
    qs = self.policy(context)
    ind = np.argmax(qs.data.numpy())
    return actions[ind]
    
  def get_state(self):
    vis = self.vision().flatten()
    smell = self.scent()
    return np.concatenate([vis, smell])

  def save(self, filepath):
    pass

  def _load(self, filepath):
    pass

if __name__ == '__main__':
  from config import *
  env0 = env.Environment(config1)
  agent = RLAgent(env0)
  embed()



      
    


