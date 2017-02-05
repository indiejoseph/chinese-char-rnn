import sys
import tensorflow as tf
import numpy as np
import math

from base import Model
from hm_rnn import HmGruCell, MultiHmRNNCell
from adaptive_softmax import adaptive_softmax_loss


class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               layer_depth=2, rnn_size=128, cell_type='LN_LSTM',
               seq_length=50, learning_rate=1, keep_prob=0.5, grad_clip=5.0, is_training=True):

    Model.__init__(self)

    self.is_training = is_training

    # RNN
    self.rnn_size = rnn_size
    self.layer_depth = layer_depth
    self.keep_prob = keep_prob

    adaptive_softmax_cutoff = [2000, vocab_size]
    adagrad_eps = 1e-5

    if cell_type == 'GRU':
      cell = tf.nn.rnn_cell.GRUCell(rnn_size)
    elif cell_type == 'LSTM':
      cell = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)
    elif cell_type == 'HM':
      cell = HmGruCell(rnn_size)
    else:
      cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)

    if is_training and keep_prob < 1:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)

    if layer_depth > 1:
      if cell_type is not 'HM':
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer_depth, state_is_tuple=True)
      else:
        self.cell = cell = MultiHmRNNCell([cell] * layer_depth, rnn_size)

    if is_training and keep_prob < 1:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")
    self.initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      self.embedding = tf.get_variable("embedding",
        initializer=tf.random_uniform([vocab_size, rnn_size], -1.0, 1.0))
      inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

    outputs, self.final_state = tf.nn.dynamic_rnn(cell,
      inputs,
      time_major=False,
      swap_memory=True,
      initial_state=self.initial_state,
      dtype=tf.float32)
    output = tf.reshape(outputs, [-1, rnn_size])

    with tf.variable_scope('softmax'):
      softmax_w = tf.transpose(self.embedding) # weight tying
      softmax_b = tf.get_variable("softmax_b", [vocab_size], initializer=tf.constant_initializer(0.0))

    with tf.variable_scope("output"):
      self.logits = tf.matmul(output, softmax_w) + softmax_b
      self.probs = tf.nn.softmax(self.logits)

    labels = tf.reshape(self.targets, [-1])

    self.loss, training_losses = adaptive_softmax_loss(output,
        labels, adaptive_softmax_cutoff)
    self.cost = tf.reduce_sum(self.loss) / batch_size
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    tvars = tf.trainable_variables()
    optimizer = tf.train.AdagradOptimizer(learning_rate, adagrad_eps)
    tvars = tf.trainable_variables()
    grads = tf.gradients([tf.reduce_sum(loss) / batch_size for loss in training_losses], tvars)
    grads = [tf.clip_by_norm(grad, grad_clip) if grad is not None else grad for grad in grads]
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


if __name__ == '__main__':
  model = CharRNN()
