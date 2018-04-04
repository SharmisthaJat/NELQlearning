import numpy as np
from collections import deque

import environment as env
import nel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from six.moves import cPickle
from torch.autograd import Variable

actions = [nel.Direction.UP, nel.Direction.DOWN, nel.Direction.LEFT,
           nel.Direction.RIGHT]
torch.set_printoptions(precision=10)


class Policy(nn.Module):
    def __init__(self, state_size, action_dim=len(actions), history_len=2,
                 hidden_size=128):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size * history_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)

class BaseAgent(nel.Agent):
    def __init__(self, env, load_filepath=None):
        super(BaseAgent, self).__init__(env.simulator, load_filepath)
        self.env = env
    
    def save(self):
        pass
    
    def _load(self):
        pass

    def next_move(self):
        pass
    
    def step(self):
        pass

class RLAgent(BaseAgent):
    def __init__(self, env, state_size, history_len=1, load_filepath=None):
        super(RLAgent, self).__init__(env, load_filepath)
        self.policy = Policy(state_size=state_size)
        self.target = Policy(state_size=state_size)
        self.target.load_state_dict(self.policy.state_dict())
        self.prev = torch.Tensor([0, 0, 0, 0])
        self.prev_action = np.zeros(len(actions), dtype=np.float32)

        for param in self.target.parameters():
            param.requires_grad = False
        self.prev_states = deque(maxlen=history_len)
        self.history_len = history_len

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def move_fn(self, epsilon):
        return lambda epsilon=epsilon: self.next_move(epsilon=epsilon)

    def next_move(self, epsilon=0.05):
        # If the current sequence of states is less than the allowed min history
        # length, we randomly pick an action and add it to the history.
        if len(self.prev_states) < self.history_len:
            self.prev_states.append(self.create_current_frame())
            # return actions[np.random.randint(0, len(actions))]
            return np.random.choice(actions) #p=[0.5, 0.1, 0.2, 0.2])

        # If we have a full history, with probability epsilon, we pick a random
        # action, in order to explore.
        random_prob = np.random.rand()
        if random_prob < epsilon:
            self.prev_states.append(self.create_current_frame())
            return np.random.choice(actions) #p=[0.5, 0.1, 0.2, 0.2])
            # return actions[np.random.randint(0, len(actions))]

        # If we don't explore, then we chose the action with max Q value.
        state = self.get_state()
        context = Variable(torch.from_numpy(state), requires_grad=False)
        self.prev_states.append(self.create_current_frame())
        qs = self.policy(context)
        self.prev = qs.data
        # Pick the argmax.
        ind = np.argmax(qs.data.numpy())
        # Alternatively, you can sample proportional to qs.
        return actions[ind]

    def create_current_frame(self):
        vis = self.vision().flatten()
        smell = self.scent()
        return np.concatenate([vis, smell, self.prev_action])

    def get_state(self):
        if len(self.prev_states) > 0:
            context = np.concatenate(self.prev_states)
        else:
            context = np.array([])
        return np.concatenate([context, self.create_current_frame()])

    def step(self, epsilon=0.05):
        current_step_and_reward = self.env.step(self, self.move_fn(epsilon))
        self.prev_action = np.zeros(len(actions), dtype=np.float32)
        self.prev_action[current_step_and_reward[0].value] = 1.0
        return current_step_and_reward

    def save(self, filepath):
        target_path = filepath+'.target'
        with open(target_path, 'wb') as f:
            torch.save(self.target, f)

        model_path = filepath+'.model'
        with open(model_path, 'wb') as f:
            torch.save(self.policy, f)

    def _load(self, filepath):
        target_path = filepath+'.target'
        with open(target_path, 'rb') as f:
            self.target = torch.load(f)

        model_path = filepath+'.model'
        with open(model_path, 'rb') as f:
            self.policy = torch.load(f)



class RandomAgent(BaseAgent):
    def __init__(self, env, load_filepath=None):
        super(RandomAgent, self).__init__(env, load_filepath)

    def next_move(self):
        return np.random.choice(actions, p=[0.5, 0.1, 0.2, 0.2])

    def step(self):
        return self.env.step(self, self.next_move)

    def save(self, filepath):
        pass

    def _load(self, filepath):
        pass


if __name__ == '__main__':
    from config import *

    env0 = env.Environment(config1)
    agent = RLAgent(env0, 30)
