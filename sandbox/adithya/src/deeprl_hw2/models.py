import math
import os

import numpy as np
import tensorflow as tf

def create_fc_layer(input, num_neurons, activation, name):
  input_shape = input.get_shape()
  with tf.name_scope(name):
    W = tf.Variable(
    tf.random_normal(
      [input_shape[-1].value, num_neurons], stddev=1 / math.sqrt(input_shape[-1].value)),
    name='W')
    b = tf.Variable(tf.zeros([num_neurons]), name='b')

  preactivation = tf.matmul(input, W) + b
  output = activation(preactivation)
  return output, preactivation, [W, b]

class Model:
  """
    Subclassess need to:
      Implement the method _get_output,
        which returns a tuple of the output action vector and the weights used.
      Set member model_name.

    Interface (members):
      x        : State input s (placeholder)
      q        : Output, Q(s, a) for all a
      weights  : All trainable (or not) Variables
                 (used for summaries in Tensorboard)

    Interface (methods):
      compile(window, input_shape, num_actions) : Builds the computation graph
                                                  and sets x, q, and weights.
      sync_target_network(sess)                 : Sets target network weights to
                                                  that of the online network.
      run_q_online(self, sess, x_batch)         : Computes Q values for all
                                                  actions using online network.
      run_q_target(self, sess, x_batch)         : Computes Q values for all
                                                  actions using target network.
  """
  def __init__(self, exp_name):
    self.exp_name = exp_name

  def _run_q(self, sess, x_batch, q_outputs):
    return sess.run(q_outputs, feed_dict={ self.x : x_batch / 255.0 })

  def run_q_online(self, sess, x_batch):
    return self._run_q(sess, x_batch, self.q)

  def run_q_target(self, sess, x_batch):
    return self._run_q(sess, x_batch, self._target_q)

  def run_q_target_online(self, sess, x_batch):
    """ Returns (target, online) tuple. """
    return self._run_q(sess, x_batch, [self._target_q, self.q])

  def sync_target_network(self, sess):
    sess.run(self._sync_ops)

  def save(self, saver, sess, step=0):
    checkpoint_dir = self.exp_name + '-checkpoints/' + str(step)
    if not os.path.exists(self.exp_name + '-checkpoints'):
      os.makedirs(self.exp_name + '-checkpoints')
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)
    print("Saved a checkpoint: %s-%d for %s" % (self.model_name, step, self.exp_name))

  def load(self, saver, sess, checkpoint_dir):
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      print("...found %s " % ckpt_name)
      saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    else:
      print("...ain't no checkpoints here!")

  def compile(self, window, input_shape, num_actions):
    self._window = window
    self._input_shape = input_shape
    self._num_actions = num_actions

    # Policy is a Single-hidden layer feedforward neural network
    with tf.name_scope(self.model_name):
      self.x = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], window], name='input')
      with tf.name_scope("online_network"):
        self.q, self._online_weights = self._get_output()

      with tf.name_scope("target_network"):
        self._target_q, self._target_weights = self._get_output()

    self.weights = self._online_weights + self._target_weights
    self._sync_ops = [target.assign(online) for target, online in zip(self._target_weights, self._online_weights)]

class Basic(Model):
  model_name = "BasicQ"
  def _get_output(self):
    hidden = tf.Variable(tf.random_normal([1, self._num_actions]), name='HIDE')
    dummy = tf.Variable(tf.random_normal([1]), name='dummy')
    return 2 * hidden, [hidden, dummy]

class LinearQN(Model):
  model_name = "LinearQ"
  def _get_output(self):
    flattened_x = tf.reshape(self.x, [-1, np.prod(self._input_shape) * self._window])
    _, output, out_vars = create_fc_layer(flattened_x, self._num_actions, tf.identity, 'output')
    return output, out_vars

class MLP(Model):
  model_name = "MLPQ"
  def _get_output(self):
    flattened_x = tf.reshape(self.x, [-1, np.prod(self._input_shape) * self._window])
    h_out, h_pre, h_vars = create_fc_layer(flattened_x, 100, tf.sigmoid, 'hidden1')
    also_q, output, out_vars = create_fc_layer(h_out, self._num_actions, tf.identity, 'output')
    weights = h_vars + out_vars
    return output, weights

class _DQN(Model):
  def _get_conv_layers_output(self):
    # TODO What type of padding to use?
    # TODO Do this in a for loop.

    # This is from Mnih's first paper.
    # The Nature paper has bigger layers (and more of them).
    kernel1 = tf.Variable(tf.random_normal([8, 8, self._window, 16], stddev=1 / math.sqrt(np.prod(self._input_shape) * self._window)), name='filter8x8')
    h1 = tf.nn.conv2d(self.x, kernel1, strides=[1, 4, 4, 1], padding="VALID")
    nl1 = tf.nn.relu(h1)

    kernel2 = tf.Variable(tf.random_normal([4, 4, 16, 32], stddev=1 / math.sqrt(np.prod(self._input_shape[:-1]) * 16)), name='filter4x4')
    h2 = tf.nn.conv2d(nl1, kernel2, strides=[1, 2, 2, 1], padding="VALID")
    nl2 = tf.nn.relu(h2)

    # Flatten the output.
    return tf.reshape(nl2, [-1, np.prod(nl2.get_shape().as_list()[1:])]), [kernel1, kernel2]

class DQN(_DQN):
  model_name = "DQN"
  def _get_output(self):
    conv_out, conv_weights = self._get_conv_layers_output()

    fc_out, _, fc_weights = create_fc_layer(conv_out, 256, tf.nn.relu, 'fc')
    output, _, final_weights = create_fc_layer(fc_out, self._num_actions, tf.identity, 'final')

    return output, conv_weights + fc_weights + final_weights

class DuelingDQN(_DQN):
  model_name = "DuelingDQN"
  def _get_output(self):
    conv_out, conv_weights = self._get_conv_layers_output()

    fc_out_v, _, fc_weights_v = create_fc_layer(conv_out, 256, tf.nn.relu, 'fc_value')
    output_v, _, final_weights_v = create_fc_layer(fc_out_v, 1, tf.identity, 'final_value')

    fc_out_a, _, fc_weights_a = create_fc_layer(conv_out, 256, tf.nn.relu, 'fc_advantage')
    output_a, _, final_weights_a = create_fc_layer(fc_out_a, self._num_actions, tf.identity, 'final_advantage')

    value_base = tf.squeeze(output_v) - tf.reduce_mean(output_a, 1)
    output = tf.expand_dims(value_base, 1) + output_a

    return output, conv_weights + fc_weights_v + final_weights_v + fc_weights_a + final_weights_a
