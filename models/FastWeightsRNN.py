"""
Fast Weights Cell.

Ba et al. Using Fast Weights to Attend to the Recent Past
https://arxiv.org/abs/1610.06258
"""

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.layers.python.layers  import layer_norm
from tensorflow.python.util import nest
import tensorflow as tf
import numpy as np

class LayerNormFastWeightsBasicRNNCell(rnn_cell.RNNCell):

  def __init__(self, num_units, forget_bias=1.0, reuse_norm=False,
               input_size=None, activation=nn_ops.relu,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               loop_steps=1, decay_rate=0.9, learning_rate=0.5,
               dropout_keep_prob=1.0, dropout_prob_seed=None):

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._reuse_norm = reuse_norm
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._S = loop_steps
    self._eta = learning_rate
    self._lambda = decay_rate
    self._g = norm_gain
    self._b = norm_shift

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope=None):
    reuse = tf.get_variable_scope().reuse
    with vs.variable_scope(scope or "Norm") as scope:
      normalized = layer_norm(inp, reuse=reuse, scope=scope)
      return normalized

  def _fwlinear(self, args, output_size, scope=None):
    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
    assert len(args) == 2
    assert args[0].get_shape().as_list()[1] == output_size

    dtype = [a.dtype for a in args][0]

    with vs.variable_scope(scope or "Linear"):
      matrixW = vs.get_variable(
        "MatrixW", dtype=dtype, initializer=tf.convert_to_tensor(np.eye(output_size, dtype=np.float32) * .05))

      matrixC = vs.get_variable(
        "MatrixC", [args[1].get_shape().as_list()[1], output_size], dtype=dtype)

      res = tf.matmul(args[0], matrixW) + tf.matmul(args[1], matrixC)
      return res

  def zero_fast_weights(self, batch_size, dtype):
    """Return zero-filled fast_weights tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      A zero filled fast_weights of shape [batch_size, state_size, state_size]
    """
    state_size = self.state_size

    zeros = array_ops.zeros(
        array_ops.pack([batch_size, state_size, state_size]), dtype=dtype)
    zeros.set_shape([None, state_size, state_size])

    return zeros

  def _vector2matrix(self, vector):
    memory_size = vector.get_shape().as_list()[1]
    return tf.reshape(vector, [-1, memory_size, 1])

  def _matrix2vector(self, matrix):
    return tf.squeeze(matrix, [2])

  def __call__(self, inputs, state, scope=None):
    state, fast_weights = state
    with vs.variable_scope(scope or type(self).__name__) as scope:
      """Compute Wh(t) + Cx(t)"""
      linear = self._fwlinear([state, inputs], self._num_units, False)
      """Compute h_0(t+1) = f(Wh(t) + Cx(t))"""
      if not self._reuse_norm:
        h = self._activation(self._norm(linear, scope="Norm0"))
      else:
        h = self._activation(self._norm(linear))
      h = self._vector2matrix(h)
      linear = self._vector2matrix(linear)
      for i in range(self._S):
        """
        Compute h_{s+1}(t+1) = f([Wh(t) + Cx(t)] + A(t) h_s(t+1)), S times.
        See Eqn (2) in the paper.
        """
        if not self._reuse_norm:
          h = self._activation(self._norm(linear +
                                          math_ops.batch_matmul(fast_weights, h), scope="Norm%d" % (i + 1)))
        else:
          h = self._activation(self._norm(linear +
                                          math_ops.batch_matmul(fast_weights, h)))

      """
      Compute A(t+1)  according to Eqn (4)
      """
      state = self._vector2matrix(state)
      new_fast_weights = self._lambda * fast_weights + self._eta * math_ops.batch_matmul(state, state, adj_y=True)

      h = self._matrix2vector(h)

      return h, (h, new_fast_weights)
