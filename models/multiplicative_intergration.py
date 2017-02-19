import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops


'''the classes here contain integrative multiplication versions of the RNN which converge faster and lead to better scores
http://arxiv.org/pdf/1606.06630v1.pdf
'''

def mi_linear(arg1, arg2, output_size, global_bias_start=0.0, scope=None):
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
