from agent import RLAgent
from environment import Environment
from config import config2
from plot import plot_reward
import nel

from collections import deque
import random
# import cPickle
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt


def center_painter_on_agent(painter, agent):
    position = agent.position()
    painter.set_viewbox(
        (position[0] - 70, position[1] - 70),
        (position[0] + 70, position[1] + 70))


def test(agent, env):
    position = agent.position()
    painter = nel.MapVisualizer(env.simulator, config2,
        (position[0] - 70, position[1] - 70),
        (position[0] + 70, position[1] + 70))
    for _ in range(1000):
        s1 = agent.get_state()
        action, reward = agent.step(epsilon=0.1)
        print(reward)
        center_painter_on_agent(painter, agent)
        painter.draw()


def main():
    env = Environment(config2)
    state_size = (config2.vision_range*2 + 1)**2 * config2.color_num_dims + config2.scent_num_dims
    agent = RLAgent(env, state_size=state_size)
    agent._load("outputs/models/NELQ_190000")
    # ._load("NELQ.model")

    # optimizer = optim.Adam(agent.policy.parameters())
    # print list(agent.policy.parameters())
    test(agent, env)


if __name__ == '__main__':
    main()
