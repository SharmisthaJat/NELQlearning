"""Main DQN agent."""

import collections
import os
import signal

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .hyperparams import WINDOW_SIZE, LOG_DIR, LR, CHECKPOINT_MAX_TO_KEEP, CHECKPOINT_SAVE_FREQ, EVAL_FREQ, EVAL_NUM_EPISODES, REPLAY_START_SIZE, MAX_REWARD_LIMIT, MIN_REWARD_LIMIT, REPORT_FREQ
from .objectives import cross_entropy, mean_huber_loss, mean_squared_error
from .core import Sample
from .policy import UniformRandomPolicy, GreedyPolicy

class DQNAgent:
  """Class implementing DQN.

  This is a basic outline of the functions/parameters you will need
  in order to implement the DQNAgnet. This is just to get you
  started. You may need to tweak the parameters, add new ones, etc.

  Feel free to change the functions and function parameters that the
  class provides.

  We have provided docstrings to go along with our suggested API.

  Parameters
  ----------
  q_network: keras.models.Model
    Your Q-network model.
  preprocessor: deeprl_hw2.core.Preprocessor
    The preprocessor class. See the associated classes for more
    details.
  memory: deeprl_hw2.core.Memory
    Your replay memory.
  gamma: float
    Discount factor.
  target_update_freq: float
    Frequency to update the target network. You can either provide a
    number representing a soft target update (see utils.py) or a
    hard target update (see utils.py and Atari paper.)
  replay_buffer_size: int
    Before you begin updating the Q-network your replay memory has
    to be filled up with some number of samples. This number says
    how many.
  train_freq: int
    How often you actually update your Q-Network. Sometimes
    stability is improved if you collect a couple samples for your
    replay memory, for every Q-network update that you run.
  batch_size: int
    How many samples in each minibatch.
  """
  def __init__(self,
         q_network,
         preprocessor,
         memory,
         policy,
         gamma,
         target_update_freq,
         replay_buffer_size,
         train_freq,
         batch_size,
         output_dir=LOG_DIR,
         double_dqn=False):
    self.q_network = q_network
    self.preprocessor_creator = preprocessor
    self.memory = memory
    self.policy = policy
    self.gamma = gamma
    self.target_update_freq = target_update_freq
    self.replay_buffer_size = replay_buffer_size
    self.train_freq = train_freq
    self.batch_size = batch_size
    self.output_dir = output_dir
    self.double_dqn = double_dqn

  def compile(self, window, input_shape, num_actions,
         model_name='q_network'):
    """Setup all of the TF graph variables/ops.

    This is inspired by the compile method on the
    keras.models.Model class.

    This is a good place to create the target network, setup your
    loss function and any placeholders you might need.

    You should use the mean_huber_loss function as your
    loss_function. You can also experiment with MSE and other
    losses.

    The optimizer can be whatever class you want. We used the
    keras.optimizers.Optimizer class. Specifically the Adam
    optimizer.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    with tf.name_scope('compile'):
      self.q_network.compile(window, input_shape, num_actions)

      self.action_taken = tf.placeholder(tf.uint8, shape=[None], name='action_taken')

      # The below seems inefficient, there will be only one nonzero entry in each row.
      # But how to index in Tensorflow ? - A.S.
      mask = tf.one_hot(self.action_taken, num_actions, 1.0, 0.0, name='q_action_mask')
      relevant_q = tf.reduce_sum(self.q_network.q * mask, axis=1)

      self.target = tf.placeholder(tf.float32, shape=[None], name='target')

      # Define loss and optimizer
      self.loss = mean_huber_loss(self.target, relevant_q)
      #optimizer = tf.train.GradientDescentOptimizer(LR)
      optimizer = tf.train.RMSPropOptimizer(LR, momentum=0.95, epsilon=0.01)
      self.train_op = optimizer.minimize(self.loss)

      # add summary ops
      tf.summary.scalar('loss', self.loss)
      for var in self.q_network.weights:
        tf.summary.histogram(var.name, var)
      self.all_summaries = tf.summary.merge_all()

    # Misc routines
    self.initializer = tf.global_variables_initializer()
    model_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='compile')
    self.saver = tf.train.Saver(model_variables, max_to_keep=CHECKPOINT_MAX_TO_KEEP)


  def update_policy(self, collect_data=False):
    """Update your policy.

    Behavior may differ based on what stage of training your
    in. If you're in training mode then you should check if you
    should update your network parameters based on the current
    step and the value you set for train_freq.

    Inside, you'll want to sample a minibatch, calculate the
    target values, update your network, and then update your
    target values.

    You might want to return the loss and other metrics as an
    output. They can help you monitor how training is going.
    """
    samples = self.memory.sample(self.batch_size)

    # Prepare x_next
    next_states = [sample.state_next for sample in samples]
    x_next = np.stack(next_states, axis=0)

    actions = np.array([sample.action for sample in samples])
    r = np.array([sample.reward for sample in samples])
    # Terminal states do not get the recurrence.
    terminal_mask = 1 - np.array([sample.is_terminal for sample in samples])

    # Prepare target
    if self.double_dqn:
      qs_for_target, qs_for_maxing = self.q_network.run_q_target_online(self.sess, x_next)
      best_actions = np.argmax(qs_for_maxing, axis=1)
      best_qs = qs_for_target[range(qs_for_target.shape[0]), best_actions]
    else:
      qs_for_target = self.q_network.run_q_target(self.sess, x_next)
      best_qs = np.max(qs_for_target, axis=1)

    target = np.array(r) + self.gamma * best_qs * terminal_mask

    # Prepare x_curr
    inputs = [sample.state_curr for sample in samples]
    x_curr = np.stack(inputs, axis=0)

    # Update weights
    eval_vars = [self.loss, self.train_op]
    if collect_data:
      eval_vars.append(self.all_summaries)

    ret_vars = self.sess.run(
        eval_vars,
        feed_dict={
            self.q_network.x: x_curr / 255.0,
            self.action_taken: actions,
            self.target: target
        })

    loss = ret_vars[0]
    if collect_data:
      summary = ret_vars[2]
    else:
      summary = None

    return loss, summary

  def signal_handler(self, signal, frame):
    self.cleanup()

  def cleanup(self):
    self.q_network.save(self.saver, self.sess, step=(self.train_iter+1))
    os._exit(0)

  def reset_env(self, env, preprocessor):
    preprocessor.reset()
    return preprocessor.process_state_for_network(env.reset())

  def fit(self, env, monitored_env, num_iterations, max_episode_length=None):
    """Fit your model to the provided environment.

    Its a good idea to print out things like loss, average reward,
    Q-values, etc to see if your agent is actually improving.

    You should probably also periodically save your network
    weights and any other useful info.

    This is where you should sample actions from your network,
    collect experience samples and add them to your replay memory,
    and update your network parameters.

    Parameters
    ----------
    env: gym.Env
      This is your Atari environment. You should wrap the
      environment using the wrap_atari_env function in the
      utils.py
    num_iterations: int
      How many samples/updates to perform.
    max_episode_length: int
      How long a single episode should last before the agent
      resets. Can help exploration.
    """
    # This function corresponds to the algorithm in the paper
    self.sess = tf.Session()
    self.sess.run(self.initializer)
    self.writer = tf.summary.FileWriter(self.output_dir, self.sess.graph)

    count_iter = 1
    self.train_iter = 1
    episo_iter = 1

    preprocessor = self.preprocessor_creator()
    processed_state_curr = self.reset_env(env, preprocessor)

    avg_loss = 0

    self.evaluate(monitored_env, EVAL_NUM_EPISODES)

    current_lives = None

    while True:
      # env.render()
      if count_iter <= REPLAY_START_SIZE:
        # Use UniformRandom initially.
        action = UniformRandomPolicy(env.action_space.n).select_action()
      elif processed_state_curr.shape[2] < WINDOW_SIZE:
        # Fire until we have at least WINDOW_SIZE frames so we can run Q network.
        # We want to FIRE to (a) initiate breakout ball and (b) maybe get points?
        action = 1
      else:
        # We pass a function here to avoid running the network if we are choosing randomly.
        q_values_f = lambda: self.q_network.run_q_online(self.sess, np.array([processed_state_curr]))[0]
        action = self.policy.select_action(q_values=q_values_f)

      # Apply the action!
      state, reward, game_over, debug_info = env.step(action)
      reward = np.clip(reward, MIN_REWARD_LIMIT, MAX_REWARD_LIMIT)
      lives = debug_info['ale.lives']

      is_terminal = game_over or (current_lives is not None and lives < current_lives)
      current_lives = lives

      processed_state_next = preprocessor.process_state_for_network(state)

      # Debugging code.
      # if episo_iter > 5:
      #  import cv2
      #  view_state = np.concatenate([processed_state_next[:, :, 0], processed_state_next[:, :, 1], processed_state_next[:, :, 2], processed_state_next[:, :, 3]], axis=1)
      #  cv2.imshow('img', view_state)
      #  cv2.waitKey(0)

      episo_iter += 1

      # Add training sample to replay memory
      if processed_state_curr.shape[2] == WINDOW_SIZE and processed_state_next.shape[2] == WINDOW_SIZE:
        s = Sample(processed_state_curr, action, processed_state_next, reward, is_terminal)
        self.memory.append(s)

      if is_terminal:
        episo_iter = 1
        preprocessor.reset()

      if game_over:
        processed_state_curr = self.reset_env(env, preprocessor)

      else:
        processed_state_curr = processed_state_next

      if count_iter % self.train_freq == 0 and count_iter > REPLAY_START_SIZE:
        loss, summary = self.update_policy(collect_data=self.train_iter % CHECKPOINT_SAVE_FREQ == 0)

        avg_loss += loss / REPORT_FREQ

        if self.train_iter % REPORT_FREQ == 0:
          print("TRAIN #{}, avg. loss {}".format(self.train_iter, avg_loss))
          avg_loss = 0

        if self.train_iter % self.target_update_freq == 0:
          self.q_network.sync_target_network(self.sess)

        if self.train_iter % CHECKPOINT_SAVE_FREQ == 0:
          print("Saving a checkpoint and writing summaries.")
          self.q_network.save(self.saver, self.sess, step=self.train_iter)
          self.writer.add_summary(summary, self.train_iter)

        if self.train_iter % EVAL_FREQ == 0:
          print("Evaluating policy for %d episodes." % EVAL_NUM_EPISODES)
          self.evaluate(monitored_env, EVAL_NUM_EPISODES)

        self.train_iter += 1

      count_iter += 1
      if self.train_iter >= num_iterations:
        break

  def evaluate(self, env, num_episodes):
    """Test your agent with a provided environment.

    You shouldn't update your network parameters here. Also if you
    have any layers that vary in behavior between train/test time
    (such as dropout or batch norm), you should set them to test.

    Basically run your policy on the environment and collect stats
    like cumulative reward, average episode length, etc.

    You can also call the render function here if you want to
    visually inspect your policy.
    """

    # We should make a new one here so we don't interfere with training.
    eval_preprocessor = self.preprocessor_creator()

    all_rewards = []
    for i in range(num_episodes):
      state = self.reset_env(env, eval_preprocessor)
      episo_iter = 1
      episo_reward = 0.0
      while 1:
        #env.render()
        if episo_iter >= WINDOW_SIZE:
          q_values = lambda: self.q_network.run_q_online(self.sess, np.array([state]))[0]
          action = GreedyPolicy(1e6).select_action(q_values=q_values)
        else:
          # Fire until we have at least WINDOW_SIZE frames so we can run Q network.
          # We want to FIRE to (a) initiate breakout ball and (b) maybe get points?
          action = 1

        frame, reward, game_over, debug_info = env.step(action)

        if game_over:
          break

        state = eval_preprocessor.process_state_for_network(frame)
        episo_iter += 1
        episo_reward += reward
      all_rewards.append(episo_reward)
    print("EVALUATION with {} episodes, mean reward {}, std {}".format(num_episodes, np.mean(all_rewards), np.std(all_rewards)))
    env._flush(force=True)
