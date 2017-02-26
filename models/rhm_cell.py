import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear


def prelu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.1

  return pos + neg


def _mi_linear(arg1, arg2, output_size, global_bias_start=0.0, scope=None):
  """Multiplicated Integrated Linear map:
  See http://arxiv.org/pdf/1606.06630v1.pdf
  A * (W[0] * arg1) * (W[1] * arg2) + (W[0] * arg1 * bias1) + (W[1] * arg2 * bias2) + global_bias.
  Args:
    arg1: batch x n, Tensor.
    arg2: batch x n, Tensor.
    output_size: int, second dimension of W[i].
  global_bias_start: starting value to initialize the global bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "MILinear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if arg1 is None:
    raise ValueError("`arg1` must be specified")
  if arg2 is None:
    raise ValueError("`arg2` must be specified")
  if output_size is None:
    raise ValueError("`output_size` must be specified")

  a1_shape = arg1.get_shape().as_list()[1]
  a2_shape = arg2.get_shape().as_list()[1]

  # Computation.
  with vs.variable_scope(scope or "MILinear"):
    matrix1 = vs.get_variable("Matrix1", [a1_shape, output_size])
    matrix2 = vs.get_variable("Matrix2", [a2_shape, output_size])
    bias1 = vs.get_variable("Bias1", [1, output_size],
                 initializer=init_ops.constant_initializer(0.5))
    bias2 = vs.get_variable("Bias2", [1, output_size],
                 initializer=init_ops.constant_initializer(0.5))
    alpha = vs.get_variable("Alpha", [output_size],
                initializer=init_ops.constant_initializer(2.0))
    arg1mul = math_ops.matmul(arg1, matrix1)
    arg2mul = math_ops.matmul(arg2, matrix2)
    res = alpha * arg1mul * arg2mul + (arg1mul * bias1) + (arg2mul * bias2)
    global_bias_term = vs.get_variable(
        "GlobalBias", [output_size],
        initializer=init_ops.constant_initializer(global_bias_start))
  return res + global_bias_term


class HighwayGRUCell(rnn.RNNCell):
  """Highway GRU Network"""

  def __init__(self, num_units,
                     num_highway_layers=3, forget_bias=0.0,
                     use_recurrent_dropout=False, dropout_keep_prob=0.90,
                     use_layer_norm=True, norm_gain=1.0, norm_shift=0.0,
                     is_training=True):
    self._num_units = num_units
    self._num_highway_layers = num_highway_layers
    self._use_recurrent_dropout = use_recurrent_dropout
    self._dropout_keep_prob = dropout_keep_prob
    self._forget_bias = forget_bias
    self._is_training = is_training
    self._use_layer_norm = use_layer_norm
    self._g = norm_gain
    self._b = norm_shift

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._g)
    beta_init = init_ops.constant_initializer(self._b)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def __call__(self, inputs, state, timestep = 0, scope=None):
    current_state = state

    for highway_layer in xrange(self._num_highway_layers):

      with tf.variable_scope('h_'+str(highway_layer)):
        with vs.variable_scope("Gates"):  # Reset gate and update gate.
          if highway_layer == 0:
            h = _mi_linear(inputs, current_state, self._num_units * 2, 1.0)
          else:
            h = _linear([current_state], self._num_units * 2, True, 1.0)

          r, u = array_ops.split(h, num_or_size_splits=2, axis=1)

          # Apply Layer Normalization to the two gates
          if self._use_layer_norm:
            r = self._norm(r, scope = 'r/')
            u = self._norm(r, scope = 'u/')

          r, u = sigmoid(r), sigmoid(u)

        with vs.variable_scope("Candidate"):
          if highway_layer == 0:
            c = tf.nn.relu(_mi_linear(inputs, r * current_state, self._num_units, self._forget_bias))
          else:
            c = tf.nn.relu(_linear([r * current_state], self._num_units, True, self._forget_bias))

          c = prelu(c)

        if self._is_training and self._use_recurrent_dropout:
          c = tf.nn.dropout(c, self._dropout_keep_prob)

      current_state = u * current_state + (1 - u) * c

    return current_state, current_state


if __name__ == "__main__":
  import numpy as np
  from tensorflow.contrib.rnn.python.ops import core_rnn

  batch = 3
  num_steps = 10
  dim = 100
  my_inputs = tf.placeholder(tf.float32, [batch, num_steps, dim], name="my_inputs")
  my_inputs = [tf.squeeze(inp) for inp in tf.split(my_inputs, num_steps, 1)]
  cell = HighwayGRUCell(dim, 3, use_recurrent_dropout=True)
  initial_state = cell.zero_state(batch, tf.float32)
  finial_state = initial_state
  output, finial_state = core_rnn.static_rnn(cell, my_inputs, finial_state)
  initializer = tf.random_uniform_initializer(-0.1, 0.1)
  data = [np.random.rand(batch, dim) for i in range(num_steps)]

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  o = sess.run(output, feed_dict={i: d for i, d in zip(my_inputs, data)})

  print(o)
