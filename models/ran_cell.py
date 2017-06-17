import collections
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from models.utils import linear

_checked_scope = core_rnn_cell_impl._checked_scope

class RANCell(RNNCell):
  """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393)."""

  def __init__(self, num_units, input_size=None, activation=tanh, normalize=False, reuse=None):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._normalize = normalize
    self._reuse = reuse

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with _checked_scope(self, scope or "ran_cell", reuse=self._reuse):
      with vs.variable_scope("gates"):
        value = tf.nn.sigmoid(linear([state, inputs], 2 * self._num_units, True, normalize=self._normalize))
        i, f = array_ops.split(value=value, num_or_size_splits=2, axis=1)

      with vs.variable_scope("candidate"):
        c = linear([inputs], self._num_units, True, normalize=self._normalize)

      new_c = i * c + f * state
      new_h = self._activation(c)

    return new_h, new_c
