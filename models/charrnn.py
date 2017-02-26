import sys
import tensorflow as tf
import numpy as np
import math

from base import Model
from tensorflow.contrib import rnn
from rhm_cell import HighwayGRUCell


class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               layer_depth=2, num_units=100,
               seq_length=50, keep_prob=0.9,
               grad_clip=5.0, num_sampled=5., is_training=True):

    Model.__init__(self)

    self.is_training = is_training

    # RNN
    self.layer_depth = layer_depth
    self.keep_prob = keep_prob
    self.batch_size = batch_size
    self.num_units = num_units
    self.seq_length = seq_length
    self.num_sampled = num_sampled

    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")

    with tf.variable_scope('rnnlm', initializer=tf.contrib.layers.xavier_initializer()):
      cell = HighwayGRUCell(num_units, layer_depth,
                            dropout_keep_prob=keep_prob,
                            use_recurrent_dropout=True,
                            is_training=is_training)

      cell = rnn.OutputProjectionWrapper(cell, num_units)

      if keep_prob < 1 and is_training:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

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

      flat_output = tf.reshape(outputs, [-1, num_units])

    with tf.variable_scope("loss", initializer=tf.contrib.layers.xavier_initializer()):
      softmax_w = tf.get_variable("softmax_w", [num_units, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])

      loss = tf.nn.nce_loss(weights=tf.transpose(softmax_w), # .T for some reason
                            biases=softmax_b,
                            inputs=flat_output,
                            labels=tf.reshape(self.targets, [-1, 1]), # Column vector
                            num_sampled=num_sampled,
                            num_classes=vocab_size)

      self.cost = tf.reduce_sum(loss) / batch_size / seq_length
      self.global_step = tf.Variable(0, name="global_step", trainable=False)

    self.lr = tf.Variable(0.0, trainable=False)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


if __name__ == "__main__":
  model = CharRNN()
