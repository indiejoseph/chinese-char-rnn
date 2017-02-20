import sys
import tensorflow as tf
import numpy as np
import math

from base import Model
from rhm_cell import HighwayGRUCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import OutputProjectionWrapper


class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               layer_depth=2, rnn_size=1000, num_units=100,
               seq_length=50, learning_rate=1, keep_prob=0.9,
               num_sampled=100, grad_clip=5.0, is_training=True):

    Model.__init__(self)

    self.is_training = is_training

    # RNN
    self.rnn_size = rnn_size
    self.layer_depth = layer_depth
    self.keep_prob = keep_prob
    self.batch_size = batch_size
    self.num_units = num_units
    self.seq_length = seq_length
    self.num_sampled = num_sampled

    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")

    with tf.variable_scope('rnnlm'):
      cell = HighwayGRUCell(rnn_size, layer_depth,
                            dropout_keep_prob=keep_prob,
                            use_recurrent_dropout=True,
                            is_training=is_training)
      cell = OutputProjectionWrapper(cell, num_units)

      with tf.device("/cpu:0"):
        stdv = np.sqrt(1. / vocab_size)
        self.embedding = tf.get_variable("embedding", [vocab_size, num_units],
                                         initializer=tf.random_uniform_initializer(-stdv, stdv))
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

        if is_training and keep_prob < 1:
          inputs = tf.nn.dropout(inputs, self.keep_prob)

    self.initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope("output"):
      outputs, self.final_state = tf.nn.dynamic_rnn(cell,
                                                    inputs,
                                                    time_major=False,
                                                    swap_memory=True,
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)

      outputs = tf.reshape(outputs, [-1, num_units])
      labels = tf.to_int64(tf.reshape(self.targets, [-1, 1]))

      # noise-contrastive estimation
      softmax_weights = tf.get_variable("softmax_weights",
                                        [vocab_size, num_units],
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))
      softmax_biases = tf.get_variable("softmax_biases", [vocab_size],
                                       initializer=tf.constant_initializer(0.0))

      (negative_samples,
       true_expected_counts,
       sampled_expected_counts) = tf.nn.learned_unigram_candidate_sampler(labels,
                                                                          1,
                                                                          num_sampled,
                                                                          False,
                                                                          vocab_size,
                                                                          seed=None,
                                                                          name=None)
      self.loss = tf.nn.sampled_softmax_loss(weights=softmax_weights,
                                             biases=softmax_biases,
                                             labels=labels,
                                             inputs=outputs,
                                             num_sampled=num_sampled,
                                             num_classes=vocab_size,
                                             num_true=1,
                                             sampled_values=(negative_samples,
                                                              true_expected_counts,
                                                              sampled_expected_counts),
                                             remove_accidental_hits=True,
                                             partition_strategy='mod',
                                             name='sampled_softmax_loss')
      self.cost = tf.reduce_mean(self.loss)
      self.global_step = tf.Variable(0, name="global_step", trainable=False)

    tvars = tf.trainable_variables()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


if __name__ == "__main__":
  model = CharRNN()
