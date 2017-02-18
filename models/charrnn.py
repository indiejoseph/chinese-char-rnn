import sys
import tensorflow as tf
import numpy as np
import math

from base import Model
from rhm_cell import HighwayGRUCell
from adaptive_softmax import adaptive_softmax_loss


class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               layer_depth=2, rnn_size=100,
               seq_length=50, learning_rate=0.2, keep_prob=0.9,
               grad_clip=5.0, is_training=True):

    Model.__init__(self)

    self.is_training = is_training

    # RNN
    self.rnn_size = rnn_size
    self.layer_depth = layer_depth
    self.keep_prob = keep_prob
    self.batch_size = batch_size
    self.seq_length = seq_length

    adaptive_softmax_cutoff = [2000, vocab_size]
    cell = HighwayGRUCell(rnn_size, layer_depth,
                          use_layer_norm=True,
                          dropout_keep_prob=keep_prob,
                          use_recurrent_dropout=is_training)

    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")
    self.initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope('rnnlm'):
      with tf.device("/cpu:0"):
        self.embedding = tf.get_variable("embedding",
          initializer=tf.random_uniform([vocab_size, rnn_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

      if is_training and keep_prob < 1:
        inputs = tf.nn.dropout(tf.tanh(inputs), self.keep_prob)

      softmax_w = tf.transpose(self.embedding) # weight tying
      softmax_b = tf.get_variable("softmax_b", [vocab_size], initializer=tf.constant_initializer(0.0))

    with tf.variable_scope("output"):
      outputs, self.final_state = tf.nn.dynamic_rnn(cell,
                                                    inputs,
                                                    time_major=False,
                                                    swap_memory=True,
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)

      output = tf.reshape(outputs, [-1, rnn_size])
      self.logits = tf.matmul(output, softmax_w) + softmax_b
      self.probs = tf.nn.softmax(self.logits)

      labels = tf.reshape(self.targets, [-1])

      self.loss, training_losses = adaptive_softmax_loss(output,
          labels, adaptive_softmax_cutoff)
      self.cost = tf.reduce_mean(
          tf.reduce_sum(tf.reshape(self.loss, [self.batch_size, -1]), 1)
        ) / self.seq_length
      self.global_step = tf.Variable(0, name="global_step", trainable=False)

    tvars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    grads = tf.gradients([tf.reduce_sum(loss) / batch_size for loss in training_losses], tvars)
    grads = [tf.clip_by_norm(grad, grad_clip) if grad is not None else grad for grad in grads]
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


if __name__ == "__main__":
  model = CharRNN()
