import tensorflow as tf
import numbers
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib import rnn
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


def zoneout(x, keep_prob, noise_shape=None, seed=None, name=None):
    """Computes zoneout (including dropout without scaling).
    With probability `keep_prob`.
    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.
    Args:
      x: A tensor.
      keep_prob: A scalar `Tensor` with the same type as x. The probability
        that each element is kept.
      noise_shape: A 1-D `Tensor` of type `int32`, representing the
        shape for randomly generated keep/drop flags.
      seed: A Python integer. Used to create random seeds. See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      name: A name for this operation (optional).
    Returns:
      A Tensor of the same shape of `x`.
    Raises:
      ValueError: If `keep_prob` is not in `(0, 1]`.
    """
    with tf.name_scope(name or "dropout") as name:
        x = ops.convert_to_tensor(x, name="x")

        if isinstance(keep_prob, numbers.Real) and 1 < keep_prob >= 0:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob,
                                        dtype=x.dtype,
                                        name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        # Do nothing if we know keep_prob == 1
        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape,
                                                 seed=seed,
                                                 dtype=x.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor
        ret.set_shape(x.get_shape())
        return 1. - ret


class QRNN_pooling(rnn.RNNCell):

    def __init__(self, out_fmaps, pool_type):
        self.__pool_type = pool_type
        self.__out_fmaps = out_fmaps

    @property
    def state_size(self):
        return self.__out_fmaps

    @property
    def output_size(self):
        return self.__out_fmaps

    def __call__(self, inputs, state, scope=None):
        """
        inputs: 2-D tensor of shape [batch_size, Zfeats + [gates]]
        """
        pool_type = self.__pool_type
        # print('QRNN pooling inputs shape: ', inputs.get_shape())
        # print('QRNN pooling state shape: ', state.get_shape())
        with tf.variable_scope(scope or "QRNN-{}-pooling".format(pool_type)):
            if pool_type == 'f':
                # extract Z activations and F gate activations
                Z, F = tf.split(1, 2, inputs)
                # return the dynamic average pooling
                output = tf.multiply(F, state) + tf.multiply(tf.subtract(1., F), Z)
                return output, output
            elif pool_type == 'fo':
                # extract Z, F gate and O gate
                Z, F, O = tf.split(inputs, 3, 1)
                new_state = tf.multiply(F, state) + tf.multiply(tf.subtract(1., F), Z)
                output = tf.multiply(O, new_state)
                return output, new_state
            elif pool_type == 'ifo':
                # extract Z, I gate, F gate, and O gate
                Z, I, F, O = tf.split(inputs, 4, 1)
                new_state = tf.multiply(F, state) + tf.multiply(I, Z)
                output = tf.multiply(O, new_state)
                return output, new_state
            else:
                raise ValueError('Pool type must be either f, fo or ifo')



class QRNNCell(object):
    """ Quasi-Recurrent Neural Network Layer
        (cf. https://arxiv.org/abs/1611.01576)
    """
    def __init__(self, out_fmaps, fwidth=2,
                 activation=tf.tanh, pool_type='fo', zoneout=0.9, infer=False,
                 bias_init_val=None,
                 name='QRNN'):
        """
        pool_type: can be f, fo, or ifo
        zoneout: > 0 means apply zoneout with p = zoneout
        bias_init_val: by default there is no bias.
        """
        self.out_fmaps = out_fmaps
        self.activation = activation
        self.name = name
        self.infer = infer
        self.pool_type = pool_type
        self.fwidth = fwidth
        self.out_fmaps = out_fmaps
        self.zoneout = zoneout
        self.bias_init_val = bias_init_val

    def __call__(self, input_):
        input_shape = input_.get_shape().as_list()
        batch_size = input_shape[0]
        fwidth = self.fwidth
        out_fmaps = self.out_fmaps
        pool_type = self.pool_type
        zoneout = self.zoneout

        with tf.variable_scope(self.name):
          # gates: list containing gate activations (num of gates depending
          # on pool_type)
          Z, gates = self.convolution(input_, fwidth, out_fmaps, pool_type,
                                      zoneout)
          # join all features (Z and gates) into Tensor at dim 2 merged
          T = tf.concat([Z] + gates, 2)
          # create the pooling layer
          pooling = QRNN_pooling(out_fmaps, pool_type)
          self.initial_state = pooling.zero_state(batch_size=batch_size,
                                                  dtype=tf.float32)
          # encapsulate the pooling in the iterative dynamic_rnn
          H, last_C = tf.nn.dynamic_rnn(pooling, T,
                                        initial_state=self.initial_state)
          self.Z = Z
          return H, last_C

    def convolution(self, input_, filter_width, out_fmaps, pool_type, zoneout_):
        """ Applies 1D convolution along time-dimension (T) assuming input
            tensor of dim (batch_size, T, n) and returns
            (batch_size, T, out_fmaps)
            zoneout: regularization (dropout) of F gate
        """
        in_shape = input_.get_shape()
        in_fmaps = in_shape[-1]
        # num_gates = len(pool_type)
        gates = []
        # pad on the left to mask the convolution (make it causal)
        pinput = tf.pad(input_, [[0, 0], [filter_width - 1, 0], [0, 0]])
        with tf.variable_scope('convolutions'):
            Wz = tf.get_variable('Wz', [filter_width, in_fmaps, out_fmaps],
                                 initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
            z_a = tf.nn.conv1d(pinput, Wz, stride=1, padding='VALID')
            if self.bias_init_val is not None:
                bz = tf.get_variable('bz', [out_fmaps],
                                     initializer=tf.constant_initializer(0.))
                z_a += bz

            z = self.activation(z_a)
            # compute gates convolutions
            for gate_name in pool_type:
                Wg = tf.get_variable('W{}'.format(gate_name),
                                     [filter_width, in_fmaps, out_fmaps],
                                     initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
                g_a = tf.nn.conv1d(pinput, Wg, stride=1, padding='VALID')
                if self.bias_init_val is not None:
                    bg = tf.get_variable('b{}'.format(gate_name), [out_fmaps],
                                         initializer=tf.constant_initializer(0.))
                    g_a += bg
                g = tf.sigmoid(g_a)
                if not self.infer and zoneout_ < 1 and gate_name == 'f':
                    print('Applying zoneout {} to gate F'.format(zoneout_))
                    # appy zoneout to F
                    g = zoneout((1. - g), zoneout_)
                    # g = 1. - tf.nn.dropout((1. - g), 1. - zoneout)
                gates.append(g)
        return z, gates
