from agent import RLAgent
from environment import Environment
from config import config2
from plot import plot_reward
import nel

from collections import deque
import random
import cPickle
import copy

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
  batch_size = 64
  training_steps = 0
  epsilon = 0.5 # 0.8
  replay = deque(maxlen=10000)
  discount_factor = .99
  eval_reward = []
  eval_steps = 1000
  max_steps = 10000
  tr_reward = 0
  #painter_tr = nel.MapVisualizer(env.simulator, config2, (-30, -30), (150, 150))
  prev_weights = agent.policy.fc3.weight
  for training_steps in range(max_steps):
    if training_steps < EPS_DECAY_START:
      epsilon = EPS_START
    elif training_steps > EPS_DECAY_END:
      epsilon = EPS_END
    else:
      epsilon = EPS_START - (EPS_START - EPS_END) * (training_steps - EPS_DECAY_START) / (EPS_DECAY_END - EPS_DECAY_START)

    
    add_to_replay = len(agent.prev_states) == 1
    s1 = agent.get_state()
    #action, reward = agent.step(0.0)
    action, reward = agent.step(epsilon)
    s2 = agent.get_state()
    tr_reward += reward
    #painter_tr.draw()
    #print(reward)
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
        #agent.policy.fc3.weight.data -= .1 * agent.policy.fc3.weight.grad.data
        #agent.policy.fc2.weight.data -= .1 * agent.policy.fc2.weight.grad.data
        #agent.policy.fc1.weight.data -= .1 * agent.policy.fc1.weight.grad.data
        optimizer.step()
        #print (agent.policy.fc3.weight.data)
        #prev_weights = agent.policy.fc3.weight.data

        if training_steps % (update_frequency * 100) == 0:
          print('step = ', training_steps)
          print("loss = ",loss.data[0])
          print("train reward = ",tr_reward)
          print('')

    if training_steps % target_update_frequency == 0:
      agent.target.load_state_dict(agent.policy.state_dict())

    #if training_steps % 20000 == 0 and training_steps > 0:
    #  env_eval = Environment(config2)
    #  agent_eval = RLAgent(env_eval)          
    #  painter = nel.MapVisualizer(env_eval.simulator, config2, (-30, -30), (150, 150))
    #  state_dictionary = copy.deepcopy(agent.policy.state_dict())
    #  agent_eval.policy.load_state_dict(state_dictionary)
    #  #agent_eval.policy.load_state_dict(agent.policy.state_dict())
    #  for i in range(100):            
    #    s1 = agent_eval.get_state()
    #    action, reward = agent_eval.step()
    #    painter.draw()

    #if training_steps % eval_frequency == 0:
    #  env_eval = Environment(config2)
    #  agent_eval = RLAgent(env_eval)          
    #  #painter = nel.MapVisualizer(env_eval.simulator, config2, (-30, -30), (150, 150))
    #  state_dictionary = copy.deepcopy(agent.policy.state_dict())
    #  agent_eval.policy.load_state_dict(state_dictionary)
    #  curr_reward = 0.0
    #  for i in range(eval_steps):            
    #    s1 = agent_eval.get_state()
    #    action, reward = agent_eval.step()
    #    curr_reward+=reward
    #    #painter.draw()
    #  print('eval reward = ', curr_reward)
    #  eval_reward.append(curr_reward)           

  painter = nel.MapVisualizer(env.simulator, config2, (-30, -30), (150, 150))
  for i in range(100):            
    s1 = agent.get_state()
    action, reward = agent.step()
    painter.draw()
    
  cPickle.dump(eval_reward,open('outputs/eval_reward.pkl','w'))
  #plot_reward(eval_reward,'RL_agent_eval')
  print(eval_reward)

  #env_eval = Environment(config2)
  #agent_eval = RLAgent(env_eval)          
  #painter = nel.MapVisualizer(env_eval.simulator, config2, (-30, -30), (150, 150))
  #agent_eval.policy.load_state_dict(agent.policy.state_dict())
  #cur_reward = 0
  #for i in range(100):            
  #  s1 = agent_eval.get_state()
  #  action, reward = agent_eval.step()
  #  print reward
  #  curr_reward+=reward
  #  painter.draw()
  #print(cur_reward)


# cumulative reward for training and test 

def main():
  env = Environment(config2)
  agent = RLAgent(env)
  
 
  optimizer = optim.Adam(agent.policy.parameters(), lr=1.0)
  #print list(agent.policy.parameters())
  train(agent, env, [0,1,2,3], optimizer)

if __name__ == '__main__':
  main()

