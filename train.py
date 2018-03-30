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


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


def compute_td_loss(batch_size, agent, replay_buffer, gamma, optimizer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = agent.policy(state)
    next_q_values = agent.target(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = F.smooth_l1_loss(q_value,  Variable(expected_q_value.data))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def plot_setup():
    # plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    p1, = ax1.plot([])
    p2, = ax2.plot([])
    ax2.set_title('loss')
    print "SETUP"
    fig.canvas.draw()

    def update(frame_idx, rewards, losses):
        p1.set_xdata(range(len(rewards)))
        p1.set_ydata(rewards)

        ax1.set_title('frame %s. reward: %s' %
                      (frame_idx, np.mean([rewards[i] for i in range(-10, 0)])))
        p2.set_xdata(range(len(losses)))
        p2.set_ydata(losses)
        ax1.set_xlim([0, len(rewards)])
        ax1.set_ylim([min(rewards), max(rewards) + 10])
        ax2.set_xlim([0, len(losses)])
        ax2.set_ylim([min(losses), max(losses)])
        print max(losses)
        ax2.set_yscale('log')
        plt.draw()
        plt.pause(0.0001)

    def save(fname):
        fig.save_fig(fname)

    return update, save
# def plot_setup():
#     fig = plt.figure(figsize=(20, 5))
#     rewards = []
#     losses = []
#     ax1 = plt.subplot(131)
#     p1 = ax1.plot(rewards)
#     ax2 = plt.subplot(132)
#     ax2.set_title('loss')
#     p2 = ax2.plot(losses)
#     def update(frame_idx, rewards, losses):
#         p1[0].set_ydata(rewards)
#         ax1.set_title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
#         p2[0].set_ydata(losses)
#         plt.draw()
#     return update


def plot(frame_idx, rewards, losses):
    # clear_output(True)
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.title('frame %s. reward: %s' %
              (frame_idx, np.mean([rewards[i] for i in range(-10, 0)])))
    plt.plot(rewards)
    plt.subplot(122)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


def get_epsilon(i, EPS_START, EPS_END, EPS_DECAY_START, EPS_DECAY_END):
    if i < EPS_DECAY_START:
        epsilon = EPS_START
    elif i > EPS_DECAY_END:
        epsilon = EPS_END
    else:
        epsilon = EPS_START - (EPS_START - EPS_END) * (i -
                                                       EPS_DECAY_START) / (EPS_DECAY_END - EPS_DECAY_START)
    return epsilon


def train(agent, env, actions, optimizer):
    EPS_START = 1.
    EPS_END = .1
    EPS_DECAY_START = 1000.
    EPS_DECAY_END = 50000.

    def eps_func(i): return get_epsilon(
        i, EPS_START, EPS_END, EPS_DECAY_START, EPS_DECAY_END)
    update_frequency = 2
    target_update_frequency = 1000
    eval_frequency = 1000
    batch_size = 64
    training_steps = 0
    replay = ReplayBuffer(10000)
    discount_factor = .99
    eval_reward = []
    eval_steps = 1000
    max_steps = 1000001
    tr_reward = 0
    agent.update_target()
    losses = []
    all_rewards = deque(maxlen=100)
    rewards = []
    plt_fn, save_fn = plot_setup()
    painter = None
    #painter_tr = nel.MapVisualizer(env.simulator, config2, (-30, -30), (150, 150))
    prev_weights = agent.policy.fc3.weight
    for training_steps in range(max_steps):
        epsilon = eps_func(training_steps)
        add_to_replay = len(agent.prev_states) == 1

        s1 = agent.get_state()
        action, reward = agent.step(epsilon)
        s2 = agent.get_state()
        tr_reward += reward
        all_rewards.append(reward)
        rewards.append(np.sum(all_rewards))
        # painter_tr.draw()
        # print(reward)

        # if training_steps % 10000 < 100:
        #     position = agent.position()
        #     if painter is None:
        #         painter = nel.MapVisualizer(env.simulator, config2, (
        #             position[0] - 70, position[1] - 70), (position[0] + 70, position[1] + 70))
        #     painter.draw()
        # else:
        #     painter = None

        if add_to_replay:
            # enum issue in server machine
            replay.push(s1, action.value, reward, s2, False)

        if training_steps % update_frequency == 0:
            if batch_size < len(replay):
                loss = compute_td_loss(
                    batch_size, agent, replay, discount_factor, optimizer)
                losses.append(loss.data[0])

        if training_steps % 200 == 0 and training_steps > 0:
            print('step = ', training_steps)
            print("loss = ", loss.data[0])
            print("train reward = ", tr_reward)
            print('')
            plt_fn(training_steps, rewards, losses)

        # if training_steps % 2000 == 0 and training_steps > 0:
        #     plot(training_steps, rewards, losses)
            # s1, action, reward, s2 = zip(*sample)
            # s1 = np.array(s1)
            # reward = np.array(reward)
            # s2 = np.array(s2)
            # action = np.array(action)
            # s1 = Variable(torch.from_numpy(s1).float())
            # q1 = agent.policy(s1)
            # q1 = q1[torch.arange(0, action.size).long(),
            #         torch.LongTensor(action)]

            # s2 = Variable(torch.from_numpy(s2).float())
            # q2 = agent.target(s2).data
            # q2, _ = torch.max(q2, 1)

            # reward = torch.FloatTensor(reward)
            # y = Variable(reward + (discount_factor * q2))

            # #huber = nn.SmoothL1Loss()
            # mse = nn.MSELoss()
            # loss = mse(q1, y)
            # optimizer.zero_grad()
            # loss.backward()
            # #agent.policy.fc3.weight.data -= .1 * agent.policy.fc3.weight.grad.data
            # #agent.policy.fc2.weight.data -= .1 * agent.policy.fc2.weight.grad.data
            # #agent.policy.fc1.weight.data -= .1 * agent.policy.fc1.weight.grad.data
            # optimizer.step()
            #print (agent.policy.fc3.weight.data)
            #prev_weights = agent.policy.fc3.weight.data

            # if training_steps % (update_frequency * 100) == 0:
            #     print('step = ', training_steps)
            #     print("loss = ", loss.data[0])
            #     print("train reward = ", tr_reward)
            #     print('')

        if training_steps % target_update_frequency == 0:
            agent.update_target()
        # if training_steps % 20000 == 0 and training_steps > 0:
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

        # if training_steps % eval_frequency == 0:
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

    position = agent.position()
    painter = nel.MapVisualizer(env.simulator, config2, (
        position[0] - 70, position[1] - 70), (position[0] + 70, position[1] + 70))
    for _ in range(100):
        s1 = agent.get_state()
        action, reward = agent.step()
        painter.draw()

    with open('outputs/eval_reward.pkl', 'w') as f:
        cPickle.dump(eval_reward, f)

    with open('outputs/train_stats.pkl', 'w') as f:
        cPickle.dump((losses, rewards), f)
    
    with open("NELQ.model", 'w') as f:
        torch.save(agent.policy, f)
    save_fn('outputs/NELQplot.png')
    # plot_reward(eval_reward,'RL_agent_eval')
    print(eval_reward)

    #env_eval = Environment(config2)
    #agent_eval = RLAgent(env_eval)
    #painter = nel.MapVisualizer(env_eval.simulator, config2, (-30, -30), (150, 150))
    # agent_eval.policy.load_state_dict(agent.policy.state_dict())
    #cur_reward = 0
    # for i in range(100):
    #  s1 = agent_eval.get_state()
    #  action, reward = agent_eval.step()
    #  print reward
    #  curr_reward+=reward
    #  painter.draw()
    # print(cur_reward)


# cumulative reward for training and test

def main():
    env = Environment(config2)
    agent = RLAgent(env)

    optimizer = optim.Adam(agent.policy.parameters())
    #print list(agent.policy.parameters())
    train(agent, env, [0, 1, 2, 3], optimizer)


if __name__ == '__main__':
    main()
