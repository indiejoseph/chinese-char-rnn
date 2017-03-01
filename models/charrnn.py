import sys
import tensorflow as tf
import numpy as np
import math

from base import Model
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib import legacy_seq2seq
from qrnn import QRNNCell

class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               num_units=300, layer_depth=2,
               seq_length=50, keep_prob=0.9,
               grad_clip=5.0, is_training=True):

    Model.__init__(self)

    self.is_training = is_training

    # RNN
    self._layer_depth = layer_depth
    self._keep_prob = keep_prob
    self._batch_size = batch_size
    self._num_units = num_units
    self._seq_length = seq_length

    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")

    with tf.variable_scope('rnnlm'):
      softmax_w = tf.get_variable("softmax_w", [num_units, vocab_size],
                                  initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
      softmax_b = tf.get_variable("softmax_b", [vocab_size])

      cell = QRNNCell(num_units, pool_type="fo", fwidth=2, infer=not is_training)

      if layer_depth > 1:
        cell = rnn.MultiRNNCell(layer_depth * [cell], state_is_tuple=True)

      cell = rnn.OutputProjectionWrapper(cell, num_units)
      self.cell = cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

      with tf.device("/cpu:0"):
        self.embedding = tf.get_variable("embedding", [vocab_size, num_units])
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
        if is_training and keep_prob < 1:
          inputs = tf.nn.dropout(inputs, keep_prob)

    self.initial_states = []
    self.final_state = []
    self.qrnns = []

    qrnn_h = inputs

    with tf.variable_scope("output"):
      for qrnn_l in range(layer_depth):
        qrnn_ = QRNNCell(num_units, pool_type='fo',
                           zoneout=0.9,
                           name='QRNN_layer{}'.format(qrnn_l),
                           infer=not is_training)
        qrnn_h, last_state = qrnn_(qrnn_h)

        if self.is_training and keep_prob < 1:
          qrnn_h_f = tf.reshape(qrnn_h, [-1, num_units])
          qrnn_h_dout = tf.nn.dropout(qrnn_h_f, keep_prob,
                                      name='dout_qrnn{}'.format(qrnn_l))
          qrnn_h = tf.reshape(qrnn_h_dout, [batch_size, -1, num_units])
        #self.last_states.append(qrnn_.last_state)
        self.final_state.append(last_state)
        self.initial_states.append(qrnn_.initial_state)
        self.qrnns.append(qrnn_)
      qrnn_h_f = tf.reshape(qrnn_h, [-1, num_units])
      output = tf.reshape(tf.concat(qrnn_h_f, 1), [-1, num_units])

    with tf.variable_scope("loss"):
      self.logits = tf.matmul(output, softmax_w) + softmax_b
      self.probs = tf.nn.softmax(self.logits)

      loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * seq_length])], vocab_size)

      self.cost = tf.reduce_sum(loss) / batch_size / seq_length
      self.global_step = tf.Variable(0, name="global_step", trainable=False)

    self.lr = tf.Variable(0.0, trainable=False)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

  def sample(self, sess, chars, vocab, UNK_ID, num=200, prime='The '):
    state = sess.run(self.cell.zero_state(1, tf.float32))
    for char in prime[:-1]:
      x = np.zeros((1, 1))
      x[0, 0] = vocab.get(char, UNK_ID)
      feed = {self.input_data: x, self.initial_state:state}
      [state] = sess.run([self.final_state], feed)

    def weighted_pick(weights):
      t = np.cumsum(weights)
      s = np.sum(weights)
      return(int(np.searchsorted(t, np.random.rand(1)*s)))

    ret = prime
    char = prime[-1]
    for _ in range(num):
      x = np.zeros((1, 1))
      x[0, 0] = vocab[char]
      feed = {self.input_data: x, self.initial_state:state}
      [probs, state] = sess.run([self.probs, self.final_state], feed)
      p = probs[0]

      sample = weighted_pick(p)

      pred = chars[sample]
      ret += pred
      char = pred
    return ret

if __name__ == "__main__":
  model = CharRNN()
