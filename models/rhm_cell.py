import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

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
                     num_highway_layers=3, forget_bias=0.0, hyper_num_units=128, hyper_embedding_size=32,
                     use_recurrent_dropout=False, dropout_keep_prob=0.90, use_layer_norm=True):
    self._num_units = num_units
    self.num_highway_layers = num_highway_layers
    self.use_recurrent_dropout = use_recurrent_dropout
    self.use_layer_norm = use_layer_norm
    self.dropout_keep_prob = dropout_keep_prob
    self.forget_bias = forget_bias
    self.hyper_num_units = hyper_num_units
    self.total_num_units = self._num_units + self.hyper_num_units
    self.hyper_cell = rnn.GRUCell(hyper_num_units)
    self.hyper_embedding_size= hyper_embedding_size
    self.hyper_output = None

  def hyper_norm(self, layer, scope="hyper"):
    zw = _linear(self.hyper_output, self.hyper_embedding_size, False, scope=scope+ "z")
    alpha = _linear(zw, self._num_units, False, scope=scope+ "alpha")
    result = alpha * layer

    return result

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self.total_num_units

  def __call__(self, inputs, state, timestep=0, scope=None):
    current_state = state[:, 0:self._num_units]
    hyper_state = state[:, self._num_units:]

    with tf.variable_scope('hyper', initializer=tf.orthogonal_initializer()):
      hyper_input = tf.concat([inputs, current_state], 1)
      self.hyper_output, hyper_state = self.hyper_cell(hyper_input, hyper_state)

    for highway_layer in xrange(self.num_highway_layers):
      with tf.variable_scope('h_'+str(highway_layer)):
        if highway_layer == 0:
          h = _mi_linear(inputs, current_state, self._num_units)
          h = self.hyper_norm(h)
        else:
          h = _linear([current_state], self._num_units, True)

        if self.use_recurrent_dropout:
          h = tf.nn.dropout(h, self.dropout_keep_prob)

        h = tf.tanh(h)

      with tf.variable_scope('t_'+str(highway_layer)):
        if highway_layer == 0:
          t = _mi_linear(inputs, current_state, self._num_units, self.forget_bias)
          t = self.hyper_norm(t)
        else:
          t = _linear([current_state], self._num_units, True, self.forget_bias)

        t = tf.sigmoid(t)

      current_state = (h - current_state) * t + current_state

    return current_state, tf.concat([current_state, hyper_state], 1)


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
