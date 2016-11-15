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

    self.cell = cell = rnn_cell.GRUCell(rnn_size)

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
        self.embedding = tf.get_variable("embedding",
                                         initializer=tf.random_uniform([vocab_size, rnn_size], -0.04, 0.04))
        inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(self.embedding, self.input_data))
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    outputs, last_state = tf.nn.rnn(self.cell, inputs, initial_state=self.initial_state)
    output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
    self.logits = tf.matmul(output, softmax_w) + softmax_b
    self.probs = tf.nn.softmax(self.logits)
    labels = tf.reshape(self.targets, [-1])
    loss = tf.nn.seq2seq.sequence_loss_by_example(
      [self.logits], [labels],
      [tf.ones([batch_size * seq_length])],
      vocab_size)
    self.cost = tf.reduce_sum(loss) / batch_size
    self.final_state = last_state

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


if __name__ == '__main__':
  model = CharRNN()
