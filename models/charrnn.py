import sys
import math
from base import Model
import tensorflow as tf
# from tensorflow.python.ops.rnn_cell import LSTMCell
from mirnn import MILSTMCell
import numpy as np


class CharRNN(Model):
  def __init__(self, sess, vocab_size, batch_size=100,
               layer_depth=2, rnn_size=128, nce_samples=10,
               seq_length=50, grad_clip=5.,
               checkpoint_dir="checkpoint", dataset_name="wiki"):

    Model.__init__(self)

    self.sess = sess
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.checkpoint_dir = checkpoint_dir
    self.dataset_name = dataset_name

    # RNN
    self.rnn_size = rnn_size
    self.layer_depth = layer_depth
    self.grad_clip = grad_clip

    with tf.variable_scope('rnnlm', initializer=tf.contrib.layers.xavier_initializer()):
      cell = MILSTMCell(rnn_size, state_is_tuple=True)

      self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer_depth, state_is_tuple=True)
      self.input_data = tf.placeholder(tf.int64, [batch_size, seq_length], name="inputs")
      self.targets = tf.placeholder(tf.int64, [batch_size, seq_length], name="targets")
      self.initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      self.embedding = tf.get_variable("embedding",
                                       initializer=tf.random_uniform([vocab_size, rnn_size], -1.0, 1.0))
      inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

    with tf.variable_scope('output', initializer=tf.contrib.layers.xavier_initializer()):
      softmax_w = tf.get_variable("softmax_w",
                                  initializer=tf.truncated_normal([vocab_size, rnn_size],
                                              stddev=1.0 / math.sqrt(rnn_size)))
      softmax_b = tf.get_variable("softmax_b",
                                  initializer=tf.constant(0.1, shape=[vocab_size]))
      outputs, self.final_state = tf.nn.dynamic_rnn(self.cell,
                                                    inputs,
                                                    time_major=False,
                                                    swap_memory=True,
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)
      outputs = tf.reshape(outputs, [-1, rnn_size])
      self.logits = tf.matmul(outputs, softmax_w, transpose_b=True) + softmax_b
      self.probs = tf.nn.softmax(self.logits)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.learning_rate = tf.Variable(0.0, trainable=False)
    train_labels = tf.reshape(self.targets, [-1, 1])
    self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=softmax_w,
                                              biases=softmax_b,
                                              inputs=outputs,
                                              labels=tf.to_int64(train_labels),
                                              num_sampled=nce_samples,
                                              num_classes=vocab_size,
                                              num_true=1,
                                              remove_accidental_hits=False,
                                              name='nce_loss'))

    tvars = tf.trainable_variables()
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    tf.scalar_summary("loss", self.loss)
    self.merged_summary = tf.merge_all_summaries()

  def sample(self, sess, chars, vocab, num=200, prime='The '):
    self.initial_state = self.cell.zero_state(1, tf.float32)

    # assign final state to rnn
    state_list = []
    for c, h in self.initial_state:
      state_list.extend([c.eval(), h.eval()])

    prime = prime.decode('utf-8')

    for char in prime[:-1]:
      x = np.zeros((1, 1))
      x[0, 0] = vocab.get(char, 0)
      feed = {self.input_data: x}
      fetchs = []
      for i in range(len(self.initial_state)):
        c, h = self.initial_state[i]
        feed[c], feed[h] = state_list[i*2:(i+1)*2]
      for c, h in self.final_state:
        fetchs.extend([c, h])
      state_list = sess.run(fetchs, feed)

    def weighted_pick(weights):
      t = np.cumsum(weights)
      s = np.sum(weights)
      return(int(np.searchsorted(t, np.random.rand(1)*s)))

    ret = prime
    char = prime[-1]

    for _ in xrange(num):
      x = np.zeros((1, 1))
      x[0, 0] = vocab.get(char, 0)
      feed = {self.input_data: x}
      fetchs = [self.probs]
      for i in range(len(self.initial_state)):
        c, h = self.initial_state[i]
        feed[c], feed[h] = state_list[i*2:(i+1)*2]
      for c, h in self.final_state:
        fetchs.extend([c, h])
      res = sess.run(fetchs, feed)
      probs = res[0]
      state_list = res[1:]
      p = probs[0]
      # sample = int(np.random.choice(len(p), p=p))
      sample = weighted_pick(p)
      pred = chars[sample]
      ret += pred
      char = pred

    return ret
