
# coding: utf-8

# In[1]:


from vizdoom import *

import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import itertools as it
import pickle
from time import time, sleep

from collections import namedtuple, deque
from copy import deepcopy
from PIL import Image
from skimage import transform, io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


# In[2]:


# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 1000
stack_size=4
# NN learning settings
batch_size = 32
update_frequency=4
seq_length = 8

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 4
resolution = (3, 60, 108)
episodes_to_watch = 10

model_savefile = "./weights_drqn.dump"
# Configuration file path
config_file_path = './defend_the_center.cfg'


# In[3]:


def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_resolution(ScreenResolution.RES_400X225)
    game.init()
    print("Doom initialized.")
    return game


# In[4]:


def preprocess(image):
    image = transform.resize(image, resolution)
    image = image.astype(np.float32)
    return torch.FloatTensor(image)

def get_current_vars(game):
    return preprocess(game.get_state().screen_buffer), torch.FloatTensor(game.get_state().game_variables)

# In[5]:


class Replay:
    curIndex = 0
    size = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.replay = []
        
    def add(self, initial_vis, initial_num, post_vis, post_num, actions, rewards, terminal):
        if self.size == self.capacity:
            self.replay[self.curIndex] = (initial_vis, initial_num, post_vis, post_num, actions, rewards, terminal)
        else:
            self.replay.append((initial_vis, initial_num, post_vis, post_num, actions, rewards, terminal))
            self.size = min(self.size + 1, self.capacity)
        self.curIndex = (self.curIndex + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.replay, batch_size)
        inds = [np.random.randint(len(x[0]) - seq_length + 1) for x in batch]
        return [[x[y][ind:ind+seq_length] for y in range(len(x))] for (x, ind) in zip(batch, inds)]


# In[6]:


class DQN(nn.Module):
    def __init__(self, actions):
        super(DQN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.sc_fc = nn.Linear(2, 64)
        self.fc1 = nn.Linear(2304 + 64, 256)
        self.lstm = nn.LSTM(256, len(actions), 1, batch_first=True)

    def forward(self, x, seq_length, batch_size):
        vis, num = x
        vis = vis.view(batch_size*seq_length, resolution[0], resolution[1], resolution[2])
        num = num.view(batch_size*seq_length, 2)
        x = F.relu(self.conv1(vis))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 2304)
        x_2 = F.relu(self.sc_fc(num))
        x = torch.cat([x, x_2], 1)
        x = F.relu(self.fc1(x))
        x = x.view(batch_size, seq_length, 256)
        return self.lstm(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[7]:


def get_q(input, dqn, seq_length, batch_size):
    vis, num = input
    vis = Variable(vis)
    num = Variable(num)
    return dqn((vis, num), seq_length, batch_size)

def get_action(input, dqn, seq_length):
    qs, _ = get_q(input, dqn, seq_length, 1)
    qs = qs[:, -1, :]
    val, ind = torch.max(qs, 1)
    return ind[-1].data.numpy()[0]


# In[8]:


def learn(game, dqn, target_dqn, actions, replay, batch_size, optimizer):
    if batch_size <= replay.size:
        sample = zip(*replay.sample(batch_size))
        vis1, num1, vis2, num2, act, reward, done = sample
        
        vis1 = torch.stack([torch.stack(vis) for vis in vis1])
        num1 = torch.stack([torch.stack(num) for num in num1])
        
        vis2 = torch.stack([torch.stack(vis) for vis in vis2])
        num2 = torch.stack([torch.stack(num) for num in num2])
        
        act = torch.LongTensor(act)
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q1, _ = get_q((vis1, num1), dqn, seq_length, batch_size)
        i1, i2 = np.indices(q1.data.shape[:2])
        q1 = q1[(i1,i2, act)]
        
        q2, _ = get_q((vis2, num2), target_dqn, seq_length, batch_size)
        q2, _  = torch.max(q2, 2)
        y = reward + discount_factor * q2 * done
        
        #Huber loss
        huber = nn.SmoothL1Loss()
        loss = huber(q1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.data[0]


# In[9]:


def add_to_replay(eps, game, dqn, actions, replay):
    game.new_episode()
    num_steps = 0
    initial_vis = []
    initial_num = []
    post_vis = []
    post_num = []
    rewards = []
    terminal = []
    actions_taken = []

    while not game.is_episode_finished():
        vis1, num1 = get_current_vars(game)
        initial_vis.append(vis1)
        initial_num.append(num1)
        action = 0
        if random.random() < eps:
            action = random.randint(0, len(actions) - 1)
        else:
            action = get_action((torch.unsqueeze(torch.stack(initial_vis[-8:]), 0), torch.unsqueeze(torch.stack(initial_num[-8:]), 0)), dqn, min(len(initial_vis), 8))

        reward = game.make_action(actions[action], frame_repeat)
        if game.is_episode_finished():
            vis2 = torch.zeros(vis1.shape)
            num2 = torch.zeros(num1.shape)
        else:
            vis2, num2 = get_current_vars(game)
        post_vis.append(vis2)
        post_num.append(num2)
        rewards.append(reward)
        terminal.append(0 if game.is_episode_finished() else 1)
        actions_taken.append(action)
    replay.add(initial_vis, initial_num, post_vis, post_num, actions_taken, rewards, terminal)


# In[10]:


def train(game, dqn, target_dqn, actions, replay, optimizer, num_episodes=0, training_step=0):
    MAX_EPISODES=8000
    MAX_STEPS = 100000
    EPS_START = 1.
    EPS_END = .1
    EPS_CONST = MAX_STEPS * .1
    EPS_DECAY = MAX_STEPS * .70
    global_training_steps = training_step
    episodes = num_episodes
    epoch = training_step / learning_steps_per_epoch
    tot_frames = learning_steps_per_epoch * update_frequency

    for i in range(100):
        add_to_replay(1, game, None, actions, replay)
    
    print 'Replay Filled'
    
    while global_training_steps < MAX_STEPS:
        print "%d EPOCH" % epoch
        epoch += 1
        epsilon = 1
        if global_training_steps < EPS_CONST:
            epsilon = EPS_START
        elif global_training_steps < EPS_DECAY:
            epsilon = EPS_START - (EPS_START - EPS_END) * (global_training_steps - EPS_CONST) / (EPS_DECAY - EPS_CONST)
        else:
            epsilon = EPS_END
        num_frames = 0
        scores = []
        losses = []
        
        while num_frames < 125:
            episodes += 1
            add_to_replay(epsilon, game, dqn, actions, replay)
            if num_frames % update_frequency == 0:
                loss = learn(game, dqn, target_dqn, actions, replay, 32, optimizer)
                losses.append(loss)
                global_training_steps += 1
            num_frames += 1
        
        target_dqn.load_state_dict(dqn.state_dict())
        minimum, maximum, mean = test(game, target_dqn, actions, visualize=False)
        print("Epoch score (min, mean, max): (%f, %f, %f)" % (minimum, mean, maximum))
        print("Mean loss: %f    Last loss: %f" % (np.mean(losses), losses[-1]))
        print("Training steps: %d" % global_training_steps)
        print("Episodes: %d" % episodes)
        torch.save(target_dqn, model_savefile)
    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")
    
    test(game, target_dqn, actions, visualize=True)


# In[11]:


a = []
a.append(torch.Tensor([1,2,3]))
a.append(torch.Tensor([1,2,3]))
a.append(torch.Tensor([1,2,3]))
a.append(torch.Tensor([3,4,5]))

a[-5:]
min(len(a), 3)


# In[12]:


def test(game, dqn, actions, visualize=False):
    # Reinitialize the game with window visible
    game.set_window_visible(visualize)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    scores = []
    vs = deque(maxlen=8)
    ns = deque(maxlen=8)
    for i in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            vis, num = get_current_vars(game)
            vs.append(vis)
            ns.append(num)
            action = get_action((torch.unsqueeze(torch.stack(vs), 0), torch.unsqueeze(torch.stack(ns), 0)), dqn, len(vs))
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            #for _ in range(frame_repeat):
            #    print game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        
        score = game.get_total_reward()
        scores.append(score)
        if visualize:
            print("Total score: ", score)
    return np.min(scores), np.max(scores), np.mean(scores)


# In[13]:


def main():
    game = initialize_vizdoom(config_file_path)
    #actions = [list(a) for a in it.product([0, 1], repeat=game.get_available_buttons_size())]
    actions = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
    replay = Replay(replay_memory_size)
    #dqn = torch.load('weights.dump')
    dqn = DQN(actions)
    target_dqn = DQN(actions)
    target_dqn.load_state_dict(dqn.state_dict())
    for param in target_dqn.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    train(game, dqn, target_dqn, actions, replay, optimizer)


# In[14]:


if __name__ == '__main__':
    main()

