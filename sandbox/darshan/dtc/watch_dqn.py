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


# In[2]:

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000
stack_size=4
# NN learning settings
batch_size = 32
update_frequency=4

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 4
resolution = (60, 80)
episodes_to_watch = 10

model_savefile = "./weights.dump"
# Configuration file path
config_file_path = './defend_the_center.cfg'


# In[3]:


# In[3]:

def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_mode(Mode.PLAYER)
    game.set_screen_resolution(ScreenResolution.RES_400X300)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.init()
    print("Doom initialized.")
    return game


# In[4]:


# In[4]:


# In[5]:

def preprocess(image):
    image = transform.resize(image, resolution)
    image = image.astype(np.float32)
    return image

def get_current_vars(game):
    return preprocess(game.get_state().screen_buffer), torch.FloatTensor(game.get_state().game_variables)

def create_state(cur_frame, frames, replace=True):
    state = np.array(frames)
    cur_state = np.tile(cur_frame, (4-state.shape[0], 1, 1))
    state = torch.Tensor(np.vstack([state, cur_state]) if state.shape[0] != 0 else cur_state)
    if replace:
        frames.append(cur_frame)
    return state


# In[5]:


# In[7]:


class DQN(nn.Module):
    def __init__(self, actions):
        super(DQN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.sc_fc = nn.Linear(2, 64)
        self.fc1 = nn.Linear(1536 + 64, 256)
        self.fc2 = nn.Linear(256, len(actions))

    def forward(self, x):
        vis, num = x
        x = F.relu(self.conv1(vis))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 1536)
        x_2 = F.relu(self.sc_fc(num))
        x = torch.cat([x, x_2], 1)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        return self.fc2(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[7]:


def get_q(input, dqn):
    vis, num = input
    vis = Variable(vis)
    num = Variable(num)
    return dqn((vis, num))

def get_action(input, dqn):
    qs = get_q(input, dqn)
    val, ind = torch.max(qs, 1)
    return ind.data.numpy()[0]


def test(game, dqn, actions, visualize=False):
    # Reinitialize the game with window visible
    game.set_window_visible(visualize)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    frames = deque(maxlen = 3)
    scores = []
    for _ in range(episodes_to_watch):
        frames.clear()
        game.new_episode()
        while not game.is_episode_finished():
            vis, num = get_current_vars(game)
            vis = create_state(vis, frames)
            best_action_index = get_action((vis.view(1, stack_size, resolution[0], resolution[1]), num.view(1, num.shape[0])), dqn)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()
                sleep(1./60.)

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        scores.append(score)
        print("Total score: ", score)
    return np.min(scores), np.max(scores), np.mean(scores)


# In[14]:


def main():
    game = initialize_vizdoom(config_file_path)
    actions = [list(a) for a in it.product([0, 1], repeat=game.get_available_buttons_size())]
    dqn = torch.load(model_savefile)
    #dqn = DQN(actions)
    test(game, dqn, actions, visualize=True)

# In[15]:

if __name__ == "__main__":
    main()

