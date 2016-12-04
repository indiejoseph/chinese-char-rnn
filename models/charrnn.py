import sys
from base import Model
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, seq2seq
import numpy as np
import math


class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               layer_depth=2, rnn_size=128,
               seq_length=50, keep_prob=0.5,
               learning_rate=0.001, grad_norm=5.0,
               checkpoint_dir="checkpoint", dataset_name="wiki", is_training=True):

    Model.__init__(self)

    self.checkpoint_dir = checkpoint_dir
    self.dataset_name = dataset_name

    # RNN
    self.rnn_size = rnn_size
    self.layer_depth = layer_depth
    self.keep_prob = keep_prob

    cell = rnn_cell.BasicLSTMCell(rnn_size, forget_bias=0.0, state_is_tuple=True)

    if is_training and keep_prob < 1:
      cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    self.cell = cell = rnn_cell.MultiRNNCell([cell] * layer_depth, state_is_tuple=True)
    self.input_data = tf.placeholder(tf.int64, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int64, [batch_size, seq_length], name="targets")
    self.initial_state = cell.zero_state(batch_size, tf.float64)

    with tf.device("/cpu:0"):
      self.embedding = tf.Variable(tf.random_uniform([vocab_size, rnn_size], -1.0, 1.0),
                                   name="embedding")
      inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

    outputs, self.final_state = tf.nn.rnn(self.cell,
                                          inputs,
                                          initial_state=self.initial_state,
                                          dtype=tf.float64)

    output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
    softmax_w = tf.Variable(tf.truncated_normal([rnn_size, vocab_size],
                                                  stddev=1.0 / math.sqrt(rnn_size)))
    softmax_b = tf.get_variable("softmax_b", [vocab_size])

    with tf.variable_scope("output"):
      self.logits = tf.matmul(output, softmax_w) + softmax_b
      self.probs = tf.nn.softmax(self.logits)

    self.loss = seq2seq.sequence_loss_by_example([self.logits],
                                                 [tf.reshape(self.targets, [-1])],
                                                 [tf.ones([batch_size * seq_length])],
                                                 vocab_size)
    self.cost = tf.reduce_mean(self.loss)
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=self.global_step)


if __name__ == '__main__':
  model = CharRNN()
