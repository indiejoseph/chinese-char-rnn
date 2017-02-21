import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear


class HighwayGRUCell(rnn.RNNCell):
  """Highway GRU Network"""

  def __init__(self, num_units,
                     num_highway_layers=3, forget_bias=0.0,
                     use_recurrent_dropout=False, dropout_keep_prob=0.90, is_training=True):
    self._num_units = num_units
    self.num_highway_layers = num_highway_layers
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob
    self.forget_bias = forget_bias
    self.is_training = is_training

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, timestep = 0, scope=None):
    current_state = state

    for highway_layer in xrange(self.num_highway_layers):

      with tf.variable_scope('h_'+str(highway_layer)):
        if highway_layer == 0:
          h = _linear([inputs, current_state], self._num_units, True)
        else:
          h = _linear([current_state], self._num_units, True)

        if self.is_training and self.use_recurrent_dropout:
          h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)

        h = tf.tanh(h)

      with tf.variable_scope('t_'+str(highway_layer)):
        if highway_layer == 0:
          t = tf.sigmoid(_linear([inputs, current_state], self._num_units, True, self.forget_bias))
        else:
          t = tf.sigmoid(_linear([current_state], self._num_units, True, self.forget_bias))

      current_state = (h - current_state) * t + current_state

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
