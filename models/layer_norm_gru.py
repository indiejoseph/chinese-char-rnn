import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

class LayerNormGRUCell(rnn.RNNCell):
  def __init__(self, num_units,
                     num_highway_layers=3,
                     use_layer_norm=True, norm_gain=1.0, norm_shift=0.0):
    self._num_units = num_units
    self._num_highway_layers = num_highway_layers
    self._use_layer_norm = use_layer_norm
    self._g = norm_gain
    self._b = norm_shift

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

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or "gru_cell"):
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        r, u = array_ops.split(
          value=_linear(
              [inputs, state], 2 * self._num_units, True, 1.0),
          num_or_size_splits=2,
          axis=1)

        # Apply Layer Normalization to the two gates
        if self._use_layer_norm:
          r = self._norm(r, scope = 'r/')
          u = self._norm(r, scope = 'u/')

        r, u = sigmoid(r), sigmoid(u)

      with vs.variable_scope("candidate"):
        c = tanh(_linear([inputs, r * state], self._num_units, True))

      new_h = u * state + (1 - u) * c

    return new_h, new_h


if __name__ == "__main__":
  import numpy as np
  from tensorflow.contrib.rnn.python.ops import core_rnn
  from tensorflow.contrib import rnn

  batch = 3
  num_steps = 10
  dim = 100
  my_inputs = tf.placeholder(tf.float32, [batch, num_steps, dim], name="my_inputs")
  my_inputs = [tf.squeeze(inp) for inp in tf.split(my_inputs, num_steps, 1)]
  cell = LayerNormGRUCell(dim)
  cell = rnn.MultiRNNCell(3 * [cell])
  initial_state = cell.zero_state(batch, tf.float32)
  finial_state = initial_state
  output, finial_state = core_rnn.static_rnn(cell, my_inputs, finial_state)
  initializer = tf.random_uniform_initializer(-0.1, 0.1)
  data = [np.random.rand(batch, dim) for i in range(num_steps)]

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  o = sess.run(output, feed_dict={i: d for i, d in zip(my_inputs, data)})

  print(o)
