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
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

# Q-learning settings
learning_rate = 0.00025
#learning_rate = 0.0025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 1000
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

model_savefile = "./dqn.dump"
# Configuration file path
config_file_path = './defend_the_center.cfg'


def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_resolution(ScreenResolution.RES_400X300)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.init()
    print("Doom initialized.")
    return game

def preprocess(image):
    image = transform.resize(image, resolution)
    image = image.astype(np.float32)
    return image

def get_current_vars(game):
    return preprocess(game.get_state().screen_buffer), game.get_state().game_variables

def create_state(cur_frame, frames, cur_num, nums, replace=True):
    state = np.array(frames)
    num_state = np.array(nums)

    cur_num_state = np.tile(cur_num, (4-num_state.shape[0], 1))
    cur_num_state = np.vstack([num_state, cur_num_state]) if num_state.shape[0] != 0 else cur_num_state
    cur_num_state = cur_num_state *[1./26., .01]

    cur_state = np.tile(cur_frame, (4-state.shape[0], 1, 1))
    
    state = torch.Tensor(np.vstack([state, cur_state]) if state.shape[0] != 0 else cur_state)
    num_state = torch.FloatTensor(cur_num_state.flatten())
    if replace:
        frames.append(cur_frame)
        nums.append(cur_num)
    return state, num_state

class Replay:
    curIndex = 0
    size = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.replay = []
        
    def add(self, s1, s2, action, reward, done):
        if done:
            s2 = (torch.zeros(s1[0].shape), torch.zeros(s1[1].shape))
        if self.size == self.capacity:
            self.replay[self.curIndex] = (s1, s2, action, reward, 0 if done else 1)
        else:
            self.replay.append((s1, s2, action, reward, 0 if done else 1))
            self.size = min(self.size + 1, self.capacity)
        self.curIndex = (self.curIndex + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.replay, batch_size)

class DQN(nn.Module):
    def __init__(self, actions):
        super(DQN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.sc_fc = nn.Linear(8, 128)
        self.fc1 = nn.Linear(1536 + 128, 512)
        #self.fc2 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(512, len(actions))

    def forward(self, x):
        vis, num = x
        x = F.relu(self.conv1(vis))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
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

def get_q(input, dqn):
    vis, num = input
    vis = Variable(vis)
    num = Variable(num)
    return dqn((vis, num))

def get_action(input, dqn):
    qs = get_q(input, dqn)
    val, ind = torch.max(qs, 1)
    return ind.data.numpy()[0]

def learn(game, dqn, target_dqn, frames, actions, replay, batch_size, optimizer):
    if batch_size <= replay.size:
        sample = zip(*replay.sample(batch_size))
        s1, s2, action, reward, done = sample
        vis1, num1 = zip(*s1)
        vis1 = torch.stack(np.array(vis1))
        num1 = torch.stack(np.array(num1))
        q1 = get_q((vis1, num1), dqn)
        action = np.array(action)
        q1 = q1[torch.arange(0, action.size).long(), torch.LongTensor(action)]
        
        vis2, num2 = zip(*s2)
        vis2 = torch.stack(np.array(vis2))
        num2 = torch.stack(np.array(num2))
        q2 = get_q((vis2, num2), target_dqn).data
        q2, _  = torch.max(q2, 1)
        #q2 = q2[torch.arange(0, action.size).long(), torch.LongTensor(action)]
        
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        y = Variable(reward + (discount_factor * q2 * done))
        
        #Huber loss
        huber = nn.SmoothL1Loss()
        loss = huber(q1, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(dqn.parameters(), 1)
        optimizer.step()
        return loss.data[0]

def add_to_replay(eps, game, dqn, frames, nums, actions, replay):
    vis1, num1 = get_current_vars(game)
    vis1, num1 = create_state(vis1, frames, num1, nums)
    action = 0
    if random.random() < eps:
        action = random.randint(0, len(actions) - 1)
    else:
        action = get_action((vis1[np.newaxis, ...], num1[np.newaxis, ...]), dqn)
    reward = game.make_action(actions[action], frame_repeat)
    if game.is_episode_finished():
        s2 = None
    else:
        vis2, num2 = get_current_vars(game)
        vis2, num2 = create_state(vis2, frames, num2, nums, replace=False)
        s2 = (vis2, num2)
    replay.add((vis1, num1), s2, action, reward, game.is_episode_finished())
    
def train(game, dqn, target_dqn, actions, replay, optimizer, num_episodes=0, training_step=0):
    NUM_EPOCHS = 20
    MAX_EPISODES=8000
    MAX_STEPS = 600000
    EPS_START = 1.
    EPS_END = .1
    EPS_CONST = MAX_STEPS * .1
    EPS_DECAY = MAX_STEPS * .70
    global_training_steps = training_step
    episodes = num_episodes
    epoch = training_step / learning_steps_per_epoch
    frames = deque(maxlen=3)
    nums = deque(maxlen=3)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100000,200000], gamma=0.3)
    
    for i in range(50):
        if game.is_episode_finished():
            game.new_episode()
            frames.clear()
            nums.clear()
        add_to_replay(1, game, None, frames, nums, actions, replay)
    
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
	tot_frames = learning_steps_per_epoch * update_frequency
        scores = []
        losses = []
        
        while num_frames < tot_frames:
            if game.is_episode_finished() or episodes == 0:
                game.new_episode()
                frames.clear()
                nums.clear()
                episodes += 1
            add_to_replay(epsilon, game, dqn, frames, nums, actions, replay)
            if num_frames % update_frequency == 0:
                loss = learn(game, dqn, target_dqn, frames, actions, replay, 32, optimizer)
                scheduler.step()
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

def test(game, dqn, actions, visualize=False):
    # Reinitialize the game with window visible
    game.set_window_visible(visualize)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    frames = deque(maxlen = 3)
    nums = deque(maxlen = 3)
    scores = []
    for _ in range(episodes_to_watch):
        frames.clear()
        nums.clear()
        game.new_episode()
        while not game.is_episode_finished():
            vis, num = get_current_vars(game)
            vis, num = create_state(vis, frames, num, nums)
            best_action_index = get_action((vis.view(1, stack_size, resolution[0], resolution[1]), num.view(1, num.shape[0])), dqn)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()
                if visualize:
                    sleep(1./60.)

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        scores.append(score)
        if visualize:
            print("Total score: ", score)
    return np.min(scores), np.max(scores), np.mean(scores)

def main():
    game = initialize_vizdoom(config_file_path)
    actions = [list(a) for a in it.product([0, 1], repeat=game.get_available_buttons_size())]
    #actions = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    replay = Replay(replay_memory_size)
    #dqn = torch.load('weights.dump')
    dqn = DQN(actions)
    target_dqn = DQN(actions)
    target_dqn.load_state_dict(dqn.state_dict())
    for param in target_dqn.parameters():
        param.requires_grad = False
    
    #optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    optimizer = optim.RMSprop(dqn.parameters(), lr=learning_rate, momentum=.95, eps=.01, alpha=.95)
    train(game, dqn, target_dqn, actions, replay, optimizer)


if __name__ == '__main__':
    main()

