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

import numpy as np
import matplotlib.pyplot as plt


def test(agent,env):
	position = agent.position()
	painter = nel.MapVisualizer(env.simulator, config2, (
        position[0] - 70, position[1] - 70), (position[0] + 70, position[1] + 70))
	for _ in range(1000):
		s1 = agent.get_state()
		action, reward = agent.step()
		print(reward)
		painter.draw()


def main():
    env = Environment(config2)
    agent = RLAgent(env)
    agent._load("outputs/models/NELQ_5000000.model")
    #._load("NELQ.model")

    #optimizer = optim.Adam(agent.policy.parameters())
    #print list(agent.policy.parameters())
    test(agent, env)


if __name__ == '__main__':
    main()