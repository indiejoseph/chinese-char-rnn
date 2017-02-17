import sys
import tensorflow as tf
import numpy as np
import math

from base import Model
from qrnn import QRNNCell
from adaptive_softmax import adaptive_softmax_loss


class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               layer_depth=2, num_units=1000, rnn_size=100,
               seq_length=50, learning_rate=0.2, keep_prob=0.5, zoneout=0.9,
               grad_clip=5.0, is_training=True):

    Model.__init__(self)

    self.is_training = is_training

    # RNN
    self.rnn_size = rnn_size
    self.num_units = num_units
    self.layer_depth = layer_depth
    self.keep_prob = keep_prob
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.zoneout = zoneout

    adaptive_softmax_cutoff = [2000, vocab_size]

    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")

    inputs = None
    self.final_states = []
    self.initial_states = []
    self.qrnns = []

    with tf.device("/cpu:0"):
      self.embedding = tf.get_variable("embedding",
        initializer=tf.random_uniform([vocab_size, num_units], -1.0, 1.0))

      chars = tf.split(1, seq_length, tf.expand_dims(self.input_data, -1))

      for char_idx in chars:
        char_embed = tf.nn.embedding_lookup(self.embedding, char_idx)

        if self.is_training and self.keep_prob < 1:
          char_embed = tf.nn.dropout(char_embed, self.keep_prob, name='dout_char_emb')

        if inputs is None:
          inputs = tf.squeeze(char_embed, [1])
        else:
          inputs = tf.concat(1, [inputs,
                                 tf.squeeze(char_embed, [1])])

    qrnn_h = inputs

    for qrnn_l in range(self.layer_depth):
      qrnn_ = QRNNCell(self.num_units, pool_type="fo",
                       zoneout=self.zoneout,
                       name="QRNN_layer{}".format(qrnn_l),
                       infer=(self.is_training==False))
      qrnn_h, last_state = qrnn_(qrnn_h)

      if self.is_training and self.keep_prob < 1:
          qrnn_h_f = tf.reshape(qrnn_h, [-1, self.num_units])
          qrnn_h_dout = tf.nn.dropout(qrnn_h_f, self.keep_prob,
                                      name="dout_qrnn{}".format(qrnn_l))
          qrnn_h = tf.reshape(qrnn_h_dout, [self.batch_size, -1, self.num_units])
      self.final_states.append(last_state)
      self.initial_states.append(qrnn_.initial_state)
      self.qrnns.append(qrnn_)

    output = tf.reshape(qrnn_h, [-1, num_units])

    with tf.variable_scope("softmax"):
      softmax_w = tf.transpose(self.embedding) # weight tying
      softmax_b = tf.get_variable("softmax_b", [vocab_size], initializer=tf.constant_initializer(0.0))

    with tf.variable_scope("output"):
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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    grads = tf.gradients([tf.reduce_sum(loss) / batch_size for loss in training_losses], tvars)
    grads = [tf.clip_by_norm(grad, grad_clip) if grad is not None else grad for grad in grads]
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


if __name__ == "__main__":
  model = CharRNN()
