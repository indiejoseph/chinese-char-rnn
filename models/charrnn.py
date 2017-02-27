import sys
import tensorflow as tf
import numpy as np
import math

from base import Model
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib import legacy_seq2seq


class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               layer_depth=2, num_units=100,
               seq_length=50, keep_prob=0.9,
               grad_clip=5.0, is_training=True):

    Model.__init__(self)

    self.is_training = is_training

    # RNN
    self.layer_depth = layer_depth
    self.keep_prob = keep_prob
    self.batch_size = batch_size
    self.num_units = num_units
    self.seq_length = seq_length

    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")

    with tf.variable_scope('rnnlm'):
      softmax_w = tf.get_variable("softmax_w", [num_units, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])

      cell = rnn.GRUCell(num_units)

      if is_training and keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

      if layer_depth > 1:
        self.cell = cell = rnn.MultiRNNCell(layer_depth * [cell])

      with tf.device("/cpu:0"):
        self.embedding = tf.get_variable("embedding", [vocab_size, num_units])
        inputs = tf.split(tf.nn.embedding_lookup(self.embedding, self.input_data), seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        if is_training and keep_prob < 1:
          inputs = [tf.nn.dropout(input_, keep_prob) for input_ in inputs]

    self.initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope("output"):
      def loop(prev, _):
        prev = tf.matmul(prev, softmax_w) + softmax_b
        prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
        return tf.nn.embedding_lookup(self.embedding, prev_symbol)
if is_training and keep_prob < 1:
  inputs = [tf.nn.dropout(input_, keep_prob) for input_ in inputs]
      outputs, self.final_state = tf.nn.dynamic_rnn(cell,
                                                    inputs,
                                                    time_major=False,
                                                    swap_memory=True,
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)
      output = tf.reshape(tf.concat(outputs, 1), [-1, num_units])

    with tf.variable_scope("loss"):
      self.logits = tf.matmul(output, softmax_w) + softmax_b
      self.probs = tf.nn.softmax(self.logits)

      loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * seq_length])], vocab_size)

      self.cost = tf.reduce_sum(loss) / batch_size / seq_length
      self.final_state = last_state
      self.global_step = tf.Variable(0, name="global_step", trainable=False)

    self.lr = tf.Variable(0.0, trainable=False)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

  def forward_model(self, sess, state, input_sample):
    '''Run a forward pass. Return the updated hidden state and the output probabilities.'''
    shaped_input = np.array([[input_sample]], np.float32)
    inputs = {self.input_data: shaped_input,
              self.initial_state: state}
    [probs, state] = sess.run([self.probs, self.final_state], feed_dict=inputs)
    return probs[0], state

if __name__ == "__main__":
  model = CharRNN()
