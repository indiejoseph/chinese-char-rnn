import sys
from base import Model
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import numpy as np


class CharRNN(Model):
  def __init__(self, vocab_size=1000, batch_size=100,
               layer_depth=2, rnn_size=128,
               seq_length=50, grad_clip=5., keep_prob=0.5,
               checkpoint_dir="checkpoint", dataset_name="wiki", infer=False):

    Model.__init__(self)

    self.batch_size = batch_size
    self.seq_length = seq_length
    self.checkpoint_dir = checkpoint_dir
    self.dataset_name = dataset_name

    # RNN
    self.rnn_size = rnn_size
    self.layer_depth = layer_depth
    self.grad_clip = grad_clip
    self.keep_prob = keep_prob

    self.cell = cell = rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)

    if not infer and keep_prob < 1:
      self.cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    self.cell = rnn_cell.MultiRNNCell([cell] * layer_depth, state_is_tuple=True)
    self.input_data = tf.placeholder(tf.int64, [batch_size, seq_length], name="inputs")
    self.targets = tf.placeholder(tf.int64, [batch_size, seq_length], name="targets")
    self.initial_state = self.cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope('rnnlm'):
      softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])

      with tf.device("/cpu:0"):
        self.embedding = tf.get_variable("embedding", [vocab_size, rnn_size],
                                         initializer=tf.truncated_normal_initializer(
                                           stddev=float(1.0 / np.sqrt(rnn_size))
                                         ))
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

    outputs, self.final_state = tf.nn.dynamic_rnn(self.cell,
                                                    inputs,
                                                    time_major=False,
                                                    swap_memory=True,
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)
    output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
    self.logits = tf.matmul(output, softmax_w) + softmax_b
    self.probs = tf.nn.softmax(self.logits)
    labels = tf.reshape(self.targets, [-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, labels)
    self.cost = tf.reduce_mean(loss)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.learning_rate = tf.Variable(0.0, trainable=False)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


if __name__ == '__main__':
  model = CharRNN()
