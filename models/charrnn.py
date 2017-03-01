import sys
import tensorflow as tf
import numpy as np
import math

from base import Model
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib import legacy_seq2seq
from adaptive_softmax import adaptive_softmax_loss
from lstm import BNLSTMCell

class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               rnn_size=1024, layer_depth=2, num_units=100,
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
    self._rnn_size = rnn_size

    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")

    with tf.variable_scope('rnnlm'):
      cell = BNLSTMCell(rnn_size, training=is_training)

      if is_training and keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

      if layer_depth > 1:
        cell = rnn.MultiRNNCell(layer_depth * [cell], state_is_tuple=True)

      cell = rnn.OutputProjectionWrapper(cell, num_units)
      self.cell = cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

      with tf.device("/cpu:0"):
        self.embedding = tf.get_variable("embedding", [vocab_size, num_units])
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
        if is_training and keep_prob < 1:
          inputs = tf.nn.dropout(inputs, keep_prob)

    self.initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope("output"):
      outputs, last_state = tf.nn.dynamic_rnn(cell,
                                              inputs,
                                              time_major=False,
                                              swap_memory=True,
                                              initial_state=self.initial_state,
                                              dtype=tf.float32)
      output = tf.reshape(tf.concat(outputs, 1), [-1, num_units])

    with tf.variable_scope("loss"):
      labels = tf.reshape(self.targets, [-1])
      cutoff = [2000, vocab_size]
      loss, training_losses = adaptive_softmax_loss(output, labels, cutoff)

      self.cost = tf.reduce_sum(loss) / batch_size / seq_length
      self.final_state = last_state
      self.global_step = tf.Variable(0, name="global_step", trainable=False)

    self.lr = tf.Variable(0.0, trainable=False)

    tvars = tf.trainable_variables()
    grads = tf.gradients([tf.reduce_sum(loss) / batch_size for loss in training_losses], tvars)
    grads = [tf.clip_by_norm(grad, grad_clip) if grad is not None else grad for grad in grads]
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

if __name__ == "__main__":
  model = CharRNN()
