import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

class HighwayGRUCell(rnn.RNNCell):
  """
  Highway GRU Network with Hyper Network
  """

  def __init__(self, num_units,
                     num_highway_layers=3, forget_bias=0.0, hyper_num_units=128, hyper_embedding_size=4,
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

  def hyper_norm(self, layer, num_units, scope="hyper_norm"):
    with tf.variable_scope(scope + '_z'):
      zw = _linear(self.hyper_output, self.hyper_embedding_size, True, 1.0, scope=scope+ "z")
    with tf.variable_scope(scope + '_alpha'):
      alpha = _linear(zw, num_units, False, scope=scope+ "alpha")
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
        with tf.variable_scope('gates_'+str(highway_layer)):
          h_bias = vs.get_variable("h_bias", [2 * self._num_units], initializer=tf.constant_initializer(1.))

          if highway_layer == 0:
            h = _linear([inputs, current_state], 2 * self._num_units, False)
          else:
            h = _linear([current_state], 2 * self._num_units, False)

          h = self.hyper_norm(h, 2 * self._num_units)
          r_bias, u_bias = array_ops.split(value=h_bias, num_or_size_splits=2, axis=0)
          r, u = array_ops.split(value=h, num_or_size_splits=2, axis=1)
          r = r + r_bias
          u = u + u_bias

        with vs.variable_scope("candidate_"+str(highway_layer)):
          c_bias = vs.get_variable("c_bias", [self._num_units], initializer=tf.constant_initializer(0.0))

          if highway_layer == 0:
            c = _linear([inputs, r * current_state], self._num_units, False)
          else:
            c = _linear([r * current_state], self._num_units, False)

          c = tf.tanh(self.hyper_norm(c, self._num_units) + c_bias)

          if self.use_recurrent_dropout:
            c = tf.nn.dropout(c, self.dropout_keep_prob)

        current_state = u * current_state + (1 - u) * c

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
