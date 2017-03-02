import sys
import tensorflow as tf
import numpy as np
import math

from base import Model
from tensorflow.contrib import legacy_seq2seq

class GCNNModel(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               context_size=50, num_layers=2, embedding_size=100,
               keep_prob=0.9, filter_size=64, filter_h=5, block_size=5,
               grad_clip=5.0, is_training=True):

    Model.__init__(self)

    filter_w = embedding_size
    self.input_data = tf.placeholder(tf.int32, [batch_size, context_size-1], name="inputs")
    self.targets = tf.placeholder(tf.int32, [batch_size, context_size-1], name="targets")

    with tf.variable_scope('gcnn'):
      softmax_w = tf.get_variable("softmax_w", [embedding_size, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])

      with tf.device("/cpu:0"):
        self.embedding = tf.get_variable("embedding", [vocab_size, embedding_size])
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
        if is_training and keep_prob < 1:
          inputs = tf.nn.dropout(inputs, keep_prob)
        mask_layer = np.ones((batch_size, context_size-1, embedding_size))
        mask_layer[:,0:filter_h/2,:] = 0
        inputs *= mask_layer

        inputs_shape = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, (inputs_shape[0], inputs_shape[1], inputs_shape[2], 1))
        h, res_input = inputs, inputs

    with tf.variable_scope("output"):
      for i in range(num_layers):
        fanin_depth = h.get_shape()[-1]
        filter_size = filter_size if i < num_layers-1 else 1
        shape = (filter_h, filter_w, fanin_depth, filter_size)

        with tf.variable_scope("layer_%d"%i):
          conv_w = self.conv_op(h, shape, "linear")
          conv_v = self.conv_op(h, shape, "gated")
          h = conv_w * tf.sigmoid(conv_v)
          if i % block_size == 0:
            h += res_input
            res_input = h
      h = tf.reshape(h, (-1, embedding_size))

    with tf.variable_scope("loss"):
      self.logits = tf.matmul(h, softmax_w) + softmax_b
      self.probs = tf.nn.softmax(self.logits)
      loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1, 1])],
                [tf.ones([batch_size * (context_size - 1)])], vocab_size)

      self.loss = tf.reduce_mean(loss)
      self.global_step = tf.Variable(0, name="global_step", trainable=False)

    self.lr = tf.Variable(0.0, trainable=False)

    trainer = tf.train.AdamOptimizer(self.lr)
    gradients = trainer.compute_gradients(self.loss)
    clipped_gradients = [(tf.clip_by_value(_[0], -grad_clip, grad_clip), _[1]) for _ in gradients]
    self.train_op = trainer.apply_gradients(clipped_gradients)
    self.perplexity = tf.exp(self.loss)

  def conv_op(self, fan_in, shape, name):
    W = tf.get_variable("%s_W"%name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
    b = tf.get_variable("%s_b"%name, shape[-1], tf.float32, tf.constant_initializer(1.0))
    return tf.add(tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME'), b)

if __name__ == "__main__":
  model = GCNN()
