import tensorflow as tf
import collections
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import layer_norm
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
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

_FastWeightTuple = collections.namedtuple("FastWeightTuple", ("h", "fw"))

class FastWeightTuple(_FastWeightTuple):
  __slots__ = ()

  @property
  def dtype(self):
    (h, fw) = self
    if not h.dtype == fw.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(h.dtype), str(fw.dtype)))
    return h.dtype


class FWGRUCell(rnn.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, step=2, learning_rate=0.5, decay_rate=0.9,
               use_layer_norm=True, activation=tanh):
    self._num_units = num_units
    self._activation = activation
    self._step = step
    self._lr = learning_rate
    self._decay_rate = decay_rate
    self._use_layer_norm = use_layer_norm

  @property
  def state_size(self):
    return FastWeightTuple(self._num_units, self._num_units * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _layer_norm(self, inp, scope=None):
    reuse = tf.get_variable_scope().reuse
    with vs.variable_scope(scope or "Norm") as scope:
      normalized = layer_norm(inp, reuse=reuse, scope=scope)
      return normalized

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or "gru_cell"):
      state, fast_weights = state
      fast_weights = tf.reshape(fast_weights, [-1, self._num_units, self._num_units])

      with vs.variable_scope("gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = array_ops.split(
            value=_linear([inputs, state], 2 * self._num_units, True, 1.0),
            num_or_size_splits=2,
            axis=1)
        r, u = sigmoid(r), sigmoid(u)

      with vs.variable_scope("candidate"):
        linear = _linear([inputs, r * state], self._num_units, True)

        if self._use_layer_norm:
          linear = self._layer_norm(linear, scope="norm_0")

        c = self._activation(linear)
        h = u * state + (1 - u) * c

      with vs.variable_scope("fast_weights"):
        linear = tf.reshape(h, [-1, self._num_units, 1])
        h = tf.reshape(h, [-1, self._num_units, 1])

        for i in range(self._step):
          h = linear + tf.matmul(fast_weights, h)

          if self._use_layer_norm:
            h = self._layer_norm(h, scope="norm_%d" % (i + 1))

          h = self._activation(h)

      state = tf.reshape(state, [-1, self._num_units, 1])
      new_fast_weights = self._decay_rate * fast_weights + self._lr * tf.matmul(state, state, adjoint_b=True)
      new_fast_weights = tf.reshape(new_fast_weights, [-1, self._num_units * self._num_units])
      new_h = tf.squeeze(h, [2])

    return new_h, FastWeightTuple(new_h, new_fast_weights)


if __name__ == "__main__":
  from tensorflow.contrib.rnn.python.ops import core_rnn

  batch = 3
  num_steps = 10
  dim = 100
  my_inputs = tf.placeholder(tf.float32, [batch, num_steps, dim], name="my_inputs")
  my_inputs = [tf.squeeze(inp) for inp in tf.split(my_inputs, num_steps, 1)]
  cell = FWGRUCell(dim, use_layer_norm=False)
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
