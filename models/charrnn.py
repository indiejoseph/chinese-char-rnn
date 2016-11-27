import sys
from base import Model
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import numpy as np
import math


class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               layer_depth=2, rnn_size=128,
               seq_length=50, keep_prob=0.5, decay_rate=0.9999,
               learning_rate=0.001, learning_rate_step=1000, nce_samples=25,
               checkpoint_dir="checkpoint", dataset_name="wiki", infer=False):

    Model.__init__(self)

    self.batch_size = batch_size
    self.seq_length = seq_length
    self.checkpoint_dir = checkpoint_dir
    self.dataset_name = dataset_name
    self.decay_rate = decay_rate
    self.learning_rate = learning_rate
    self.learning_rate_step = learning_rate_step
    self.nce_samples = nce_samples

    # RNN
    self.rnn_size = rnn_size
    self.layer_depth = layer_depth
    self.keep_prob = keep_prob

    self.cell = cell = rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)

    if not infer and keep_prob < 1:
      self.cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    self.cell = rnn_cell.MultiRNNCell([cell] * layer_depth, state_is_tuple=True)
    self.input_data = tf.placeholder(tf.int64, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int64, [batch_size, seq_length], name="targets")
    self.initial_state = self.cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope('rnnlm'):
      softmax_w = tf.Variable(tf.truncated_normal([vocab_size, rnn_size],
                              stddev=1.0 / math.sqrt(rnn_size)))
      softmax_b = tf.get_variable("softmax_b", [vocab_size])

      with tf.device("/cpu:0"):
        init_width = 0.5 / rnn_size
        self.embedding = tf.get_variable("embedding",
                                         initializer=tf.random_uniform([vocab_size, rnn_size], -init_width, init_width))
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

    outputs, self.final_state = tf.nn.dynamic_rnn(self.cell,
                                                  inputs,
                                                  time_major=False,
                                                  swap_memory=True,
                                                  initial_state=self.initial_state,
                                                  dtype=tf.float32)
    outputs = tf.reshape(outputs, [-1, rnn_size])
    labels = tf.reshape(self.targets, [-1, 1])

    self.logits = tf.matmul(outputs, softmax_w, transpose_b=True) + softmax_b
    self.probs = tf.nn.softmax(self.logits)
    self.loss = tf.nn.nce_loss(softmax_w,
                               softmax_b,
                               outputs,
                               tf.to_int64(labels),
                               nce_samples,
                               vocab_size)
    self.cost = tf.reduce_mean(self.loss)
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    tvars = tf.trainable_variables()
    lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.learning_rate_step,
                                    self.decay_rate, staircase=True)

    self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost, var_list=tvars,
                                                        global_step=self.global_step)


if __name__ == '__main__':
  model = CharRNN()
