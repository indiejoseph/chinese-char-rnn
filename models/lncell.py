import tensorflow as tf

class LayerNormalizedLSTMCell(tf.nn.rnn_cell.RNNCell):
  """
  Adapted from TF's BasicLSTMCell to use Layer Normalization.
  Note that state_is_tuple is always True.
  """

  def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._activation = activation

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):
      c, h = state

      # change bias argument to False since LN will add bias via shift
      concat = tf.nn.rnn_cell._linear([inputs, h], 4 * self._num_units, False)

      i, j, f, o = tf.split(1, 4, concat)

      # add layer normalization to each gate
      i = ln(i, scope = 'i/')
      j = ln(j, scope = 'j/')
      f = ln(f, scope = 'f/')
      o = ln(o, scope = 'o/')

      new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
           self._activation(j))

      # add layer_normalization in calculation of new hidden state
      new_h = self._activation(ln(new_c, scope = 'new_h/')) * tf.nn.sigmoid(o)
      new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

      return new_h, new_state


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
