"""Loss functions."""

import tensorflow as tf

def cross_entropy(y_true, y_pred):
  """Calculate the cross entropy loss.

  Parameters
  ----------
  y_true: np.array, tf.Tensor
    Target value.
  y_pred: np.array, tf.Tensor
    Predicted value.

  Returns
  -------
  tf.Tensor
    The cross entropy loss.
  """
  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(
      y_true * tf.log(y_pred), reduction_indices=[1]))
  return cross_entropy

def mean_squared_error(y_true, y_pred):
    """Calculate the mean-squared error
    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.

    Returns
    -------
    tf.Scalar
      The mse loss
    """
    error = y_true - y_pred
    # Should the below be reduce_mean? - A.S.
    return tf.reduce_sum(tf.square(error))

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    error = y_true - y_pred
    return tf.where(tf.abs(error) < max_grad / 2,
                   tf.square(error),
                   max_grad * tf.abs(error) - (max_grad ** 2.0 / 4.0))


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad))
