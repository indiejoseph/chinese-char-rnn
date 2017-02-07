from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

"""
Hierarchical Multiscale Recurrent Neural Networks (cf. https://arxiv.org/pdf/1609.01704.pdf)
Gated Feedback Recurrent Neural Networks (cf. https://arxiv.org/pdf/1502.02367)
"""

import math
import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

_linear = tf.nn.seq2seq.linear

class AttnGRUCell(tf.nn.rnn_cell.RNNCell):
  """Attention-based Gated Recurrent Unit cell (cf. https://arxiv.org/abs/1603.01417)."""

  def __init__(self, num_units):
    self._num_units = num_units

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, attention, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope(scope or 'AttnGRUCell'):
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset.
        r = sigmoid(_linear([inputs, state], self._num_units, True, 1.0))

      with tf.variable_scope("Candidate"):
        c = tanh(_linear([inputs, r * state], self._num_units, True))

      new_h = attention * c + (1 - attention) * state

    return new_h, new_h

class PredictiveMultiRNNCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self, cells, state_is_tuple=True, keep_prob=0.9):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.
    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    self._cells = cells
    self._state_is_tuple = state_is_tuple
    self._keep_prob = keep_prob
    if not state_is_tuple:
      if any(nest.is_sequence(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s"
                          % str([c.state_size for c in self._cells]))
    if not cells:
      raise ValueError("Must specify at least one cell for PredictiveMultiRNNCell.")

    for cell in cells:
      if not isinstance(cell, AttnGRUCell):
        raise TypeError("The parameter cell is not Attention GRU Cell.")

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      new_h_list = []

      if len(self._cells) > 1:
        h_prev_top = state[1]
      else:
        h_prev_top = tf.zeros_like(state[0])

      for i, cell in enumerate(self._cells):
        with vs.variable_scope("cell_%d" % i):
          if self._state_is_tuple:
            if not nest.is_sequence(state):
              raise ValueError(
                  "Expected state to be a tuple of length %d, but received: %s"
                  % (len(self.state_size), state))
            cur_state = state[i]
          else:
            cur_state = array_ops.slice(
                state, [0, cur_state_pos], [-1, cell.state_size])
            cur_state_pos += cell.state_size

          if i < len(self._cells) - 1:
            # Next cell is not the top one.
            h_prev_top = state[i+1]
          else:
            # The next cell is the top one, so give it zeros for its h_prev_top input.
            h_prev_top = tf.zeros_like(state[i])

          gating_unit_weight = vs.get_variable("w{}".format(i), h_prev_top.get_shape(), dtype=tf.float32)
          attention = sigmoid(tf.reduce_sum(gating_unit_weight * h_prev_top))

          new_h, new_state = cell(cur_inp, cur_state, attention)

          if self._keep_prob < 1:
            new_h = tf.nn.dropout(new_h, self._keep_prob)

          new_h_list.append(new_h)
          new_states.append(new_state)

    new_states = (tuple(new_states) if self._state_is_tuple
                  else array_ops.concat(1, new_states))

    return new_h_list, new_states


if __name__ == "__main__":
  import numpy as np

  batch = 3
  num_steps = 10
  dim = 2
  inputs = tf.placeholder(tf.float32, [batch, num_steps, dim], name="inputs")
  cell = AttnGRUCell(100)
  cell = PredictiveMultiRNNCell([cell] * 6)
  initial_state = cell.zero_state(batch, tf.float32)
  state = initial_state

  outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=state)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  outputs = sess.run(outputs, feed_dict={ inputs: np.random.rand(batch, num_steps, dim) })
