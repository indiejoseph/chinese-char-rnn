from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

def layer_norm(input_tensor, num_variables_in_tensor = 1, initial_bias_value = 0.0, scope = "layer_norm"):
  with tf.variable_scope(scope):
    '''for clarification of shapes:
    input_tensor = [batch_size, num_neurons]
    mean = [batch_size]
    variance = [batch_size]
    alpha = [num_neurons]
    bias = [num_neurons]
    output = [batch_size, num_neurons]
    '''
    input_tensor_shape_list = input_tensor.get_shape().as_list()

    num_neurons = input_tensor_shape_list[1]/num_variables_in_tensor



    alpha = tf.get_variable('layer_norm_alpha', [num_neurons * num_variables_in_tensor],
            initializer = tf.constant_initializer(1.0))

    bias = tf.get_variable('layer_norm_bias', [num_neurons * num_variables_in_tensor],
            initializer = tf.constant_initializer(initial_bias_value))

    if num_variables_in_tensor == 1:
      input_tensor_list = [input_tensor]
      alpha_list = [alpha]
      bias_list = [bias]

    else:
      input_tensor_list = tf.split(1, num_variables_in_tensor, input_tensor)
      alpha_list = tf.split(0, num_variables_in_tensor, alpha)
      bias_list = tf.split(0, num_variables_in_tensor, bias)

    list_of_layer_normed_results = []
    for counter in xrange(num_variables_in_tensor):
      mean, variance = moments_for_layer_norm(input_tensor_list[counter], axes = [1], name = "moments_loopnum_"+str(counter)+scope) #average across layer

      output = (alpha_list[counter] * (input_tensor_list[counter] - mean)) / variance + bias_list[counter]

      list_of_layer_normed_results.append(output)

    if num_variables_in_tensor == 1:
      return list_of_layer_normed_results[0]
    else:
      return tf.concat(1, list_of_layer_normed_results)

def moments_for_layer_norm(x, axes = 1, name = None, epsilon = 0.001):
  '''output for mean and variance should be [batch_size]'''

  if not isinstance(axes, list): axes = list(axes)

  with tf.name_scope("moments", name, [x, axes]):
    mean = tf.reduce_mean(x, axes, keep_dims = True)

    variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keep_dims = True) + epsilon)

    return mean, variance

class HighwayGRUCell(rnn.RNNCell):
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
        r, u = array_ops.split(_linear([inputs, new_h], 2 * self._num_units, True, 1.0), 2, 1)
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

      new_h = u * new_h + (1 - u) * c

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
