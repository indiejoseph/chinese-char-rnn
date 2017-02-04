from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell

_linear = tf.nn.seq2seq.linear


class LayerNormHighwayRNNCell(rnn_cell.RNNCell):
  """Highway RNN Network"""

  def __init__(self, num_units, num_highway_layers = 3, use_inputs_on_each_layer = False):
    self._num_units = num_units
    self.num_highway_layers = num_highway_layers
    self.use_inputs_on_each_layer = use_inputs_on_each_layer


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
      with tf.variable_scope('highway_factor_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          highway_factor = tf.tanh(ln(_linear([inputs, current_state], self._num_units, True)))
        else:
          highway_factor = tf.tanh(ln(_linear([current_state], self._num_units, True)))

      with tf.variable_scope('gate_for_highway_factor_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          gate_for_highway_factor = tf.sigmoid(_linear([inputs, current_state], self._num_units, True, -3.0))
        else:
          gate_for_highway_factor = tf.sigmoid(_linear([current_state], self._num_units, True, -3.0))

        gate_for_hidden_factor = 1.0 - gate_for_highway_factor

      current_state = highway_factor * gate_for_highway_factor + current_state * gate_for_hidden_factor

    return current_state, current_state

def ln(tensor, scope = None, epsilon = 1e-5):
  """ Layer normalizes a 2D tensor along its second axis """
  assert(len(tensor.get_shape()) == 2)
  m, v = tf.nn.moments(tensor, [1], keep_dims=True)
  if not isinstance(scope, str):
    scope = ''
  with tf.variable_scope(scope + 'layer_norm'):
    scale = tf.get_variable('scale',
                            shape=[tensor.get_shape()[1]],
                            initializer=tf.constant_initializer(1))
    shift = tf.get_variable('shift',
                            shape=[tensor.get_shape()[1]],
                            initializer=tf.constant_initializer(0))
  LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

  return LN_initial * scale + shift
