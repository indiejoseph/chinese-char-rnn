import sys
import tensorflow as tf
import numpy as np
import math

from base import Model
from tensorflow.contrib import rnn
from adaptive_softmax import adaptive_softmax_loss


class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               layer_depth=2, rnn_size=1000, num_units=100,
               seq_length=50, learning_rate=1, keep_prob=0.9,
               grad_clip=5.0, is_training=True):

    Model.__init__(self)

    self.is_training = is_training

    # RNN
    self.rnn_size = rnn_size
    self.layer_depth = layer_depth
    self.keep_prob = keep_prob
    self.batch_size = batch_size
    self.num_units = num_units
    self.seq_length = seq_length
    self.adaptive_softmax_cutoff = [2000, vocab_size]

    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")

    with tf.variable_scope('rnnlm'):
      cell = rnn.GRUCell(rnn_size)

      if is_training and keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, keep_prob)

      if layer_depth > 1:
        cell = rnn.MultiRNNCell([cell] * layer_depth)

      cell = rnn.OutputProjectionWrapper(cell, num_units)

      with tf.device("/cpu:0"):
        stdv = np.sqrt(1. / vocab_size)
        self.embedding = tf.get_variable("embedding", [vocab_size, num_units],
                                         initializer=tf.random_uniform_initializer(-stdv, stdv))
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

    self.initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope("output"):
      outputs, self.final_state = tf.nn.dynamic_rnn(cell,
                                                    inputs,
                                                    time_major=False,
                                                    swap_memory=True,
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)

      outputs = tf.reshape(tf.concat(outputs, 1), [-1, num_units])
      labels = tf.reshape(self.targets, [-1])

      self.loss, _ = adaptive_softmax_loss(outputs, labels, self.adaptive_softmax_cutoff)
      self.cost = tf.reduce_mean(self.loss)
      self.global_step = tf.Variable(0, name="global_step", trainable=False)

    tvars = tf.trainable_variables()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


if __name__ == "__main__":
  model = CharRNN()
