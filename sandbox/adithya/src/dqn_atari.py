#!/usr/bin/env python
"""Run Atari Environment with DQN."""

import argparse
import os
import signal
import sys

import gym

from deeprl_hw2.core import ExperienceReplay
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.hyperparams import *
from deeprl_hw2.models import Basic, DQN, MLP, DuelingDQN, LinearQN
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.policy import UniformRandomPolicy, GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from deeprl_hw2.preprocessors import AtariPreprocessor, PreprocessorSequence, HistoryPreprocessor

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir

def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvadersDeterministic-v3', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--model', default='dqn', help='Q Network type to use.')
    parser.add_argument('--double', action='store_true')

    model_map = { 'linear'  : LinearQN,
                  'mlp'     : MLP,
                  'dqn'     : DQN,
                  'dueling' : DuelingDQN
    }

    args = parser.parse_args()

    args.model = args.model.lower()
    if args.model not in model_map:
      print("Invalid model type. Valid types are", model_map.keys())
      sys.exit(1)

    args.output = get_output_folder(args.output, args.env)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    env = gym.make(args.env)

    monitored_env = gym.wrappers.Monitor(gym.make(args.env), args.output, video_callable=lambda i: i % EVAL_NUM_EPISODES == 0)

    atari = not args.env.startswith("CartPole")

    if atari:
      input_shape = (IMAGE_SIZE, IMAGE_SIZE)
      preprocessor = lambda: PreprocessorSequence(AtariPreprocessor(new_size=input_shape), HistoryPreprocessor(history_length=WINDOW_SIZE, max_over=True))
    else:
      input_shape = (4,)
      preprocessor = lambda: HistoryPreprocessor(history_length=WINDOW_SIZE)

    memory = ExperienceReplay(max_size=REPLAY_BUFFER_SIZE, window_length=WINDOW_SIZE)

    NUM_ACTIONS = env.action_space.n
    #policy = UniformRandomPolicy(num_actions=NUM_ACTIONS)
    #policy = GreedyEpsilonPolicy(NUM_ACTIONS, EPSILON)
    policy = LinearDecayGreedyEpsilonPolicy(NUM_ACTIONS, 1.0, EPSILON,
                                            NUM_ITERATIONS_LINEAR_DECAY)

    model = model_map[args.model](exp_name=args.output)

    agent = DQNAgent(
        q_network=model,
        preprocessor=preprocessor,
        memory=memory,
        policy=policy,
        gamma=GAMMA,
        target_update_freq=TARGET_UPDATE_FREQ,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        train_freq=TRAIN_FREQ,
        batch_size=BATCH_SIZE,
        output_dir=args.output,
        double_dqn=args.double)

    agent.compile(window=WINDOW_SIZE,
        input_shape=input_shape,
        num_actions=NUM_ACTIONS,
        model_name='q_network')

    signal.signal(signal.SIGINT, agent.signal_handler)
    signal.signal(signal.SIGTERM, agent.signal_handler)
    signal.signal(signal.SIGHUP, agent.signal_handler)

    agent.fit(env, monitored_env,
        num_iterations=NUM_ITERATIONS)


if __name__ == '__main__':
    main()
