from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.python.ops.nn import rnn_cell
from tensorflow.python.ops import array_ops

_linear = tf.nn.seq2seq.linear

# support functions for layer norm
def moments_for_layer_norm(x, axes=1, name=None):
  #output for mean and variance should be [batch_size]
  # from https://github.com/LeavesBreathe/tensorflow_with_latest_papers
  epsilon = 1e-3 # found this works best.
  if not isinstance(axes, list): axes = list(axes)
  with tf.name_scope("moments", name, [x, axes]):
    mean = tf.reduce_mean(x, axes, keep_dims=True)
    variance = tf.sqrt(tf.reduce_mean(tf.square(x-mean), axes, keep_dims=True)+epsilon)
    return mean, variance

def layer_norm(input_tensor, scope="layer_norm", alpha_start=1.0, bias_start=0.0):
  # derived from:
  # https://github.com/LeavesBreathe/tensorflow_with_latest_papers, but simplified.
  with tf.variable_scope(scope):
    input_tensor_shape_list = input_tensor.get_shape().as_list()
    num_units = input_tensor_shape_list[1]

    alpha = tf.get_variable('layer_norm_alpha', [num_units],
      initializer=tf.constant_initializer(alpha_start))
    bias = tf.get_variable('layer_norm_bias', [num_units],
      initializer=tf.constant_initializer(bias_start))

    mean, variance = moments_for_layer_norm(input_tensor,
      axes=[1], name = "moments_"+scope)
    output = (alpha * (input_tensor-mean))/(variance)+bias

  return output


class HighwayGRUCell(rnn_cell.RNNCell):
  """Highway GRU Network"""

  def __init__(self, num_units,
                     num_highway_layers=3, use_inputs_on_each_layer=False,
                     use_layer_norm=True,
                     use_recurrent_dropout=False, dropout_keep_prob=0.90):
    self._num_units = num_units
    self.num_highway_layers = num_highway_layers
    self.use_inputs_on_each_layer = use_inputs_on_each_layer
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob
    self.use_layer_norm = use_layer_norm

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
    new_h = state

    for highway_layer in xrange(self.num_highway_layers):

      with tf.variable_scope('gates_'+str(highway_layer)):
        r, u = array_ops.split(1, 2, _linear([inputs, new_h],
                                             2 * self._num_units, True, 1.0))
        if self.use_layer_norm:
          r = layer_norm(r, scope="reset")
          u = layer_norm(u, scope="update")

        r, u = sigmoid(r), sigmoid(u)

      with tf.variable_scope('candidate_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          c = _linear([inputs, new_h], self._num_units, True)
        else:
          c = _linear([new_h], self._num_units, True)

        if self.use_layer_norm:
          c = layer_norm(c)

        if self.use_recurrent_dropout:
          c = tf.nn.dropout(tf.tanh(c), self.dropout_keep_prob)
        else:
          c = tf.tanh(c)

      new_h = u * state + (1 - u) * c

    return new_h, new_h


if __name__ == "__main__":
  import numpy as np

  batch = 3
  num_steps = 10
  dim = 100
  my_inputs = tf.placeholder(tf.float32, [batch, num_steps, dim], name="my_inputs")
  my_inputs = [tf.squeeze(inp) for inp in tf.split(1, num_steps, my_inputs)]
  cell = HighwayGRUCell(dim, 3, use_inputs_on_each_layer=True, use_recurrent_dropout=True)
  initial_state = cell.zero_state(batch, tf.float32)
  finial_state = initial_state
  output, finial_state = tf.nn.rnn(cell, my_inputs, finial_state)
  initializer = tf.random_uniform_initializer(-0.1, 0.1)
  data = [np.random.rand(batch, dim) for i in range(num_steps)]

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  o = sess.run(output, feed_dict={i: d for i, d in zip(my_inputs, data)})

  print(o)
