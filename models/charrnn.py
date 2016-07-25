import sys
from base import Model
import tensorflow as tf
import numpy as np
import mi_rnn_cell
from tensorflow.python.ops import rnn_cell, seq2seq

class CharRNN(Model):
  def __init__(self, sess, vocab_size, batch_size=100,
               rnn_size=512, layer_depth=2, edim=128,
               model="gru", use_peepholes=True, seq_length=50, grad_clip=5., keep_prob=0.5,
               checkpoint_dir="checkpoint", dataset_name="wiki", infer=False):

    Model.__init__(self)

    if infer:
      batch_size = 1
      seq_length = 1

    self.sess = sess

    self.batch_size = batch_size
    self.seq_length = seq_length
    self.checkpoint_dir = checkpoint_dir
    self.dataset_name = dataset_name

    # RNN
    self.model = model
    self.rnn_size = rnn_size
    self.edim = edim
    self.layer_depth = layer_depth
    self.grad_clip = grad_clip
    self.keep_prob = keep_prob

    if model == "rnn":
      cell_fn = rnn_cell.BasicRNNCell
    elif model == "gru":
      cell_fn = mi_rnn_cell.MIGRUCell
    elif model == "lstm":
      cell_fn = mi_rnn_cell.MILSTMCell
    else:
      raise Exception("model type not supported: {}".format(model))

    if model == "lstm" and use_peepholes:
      cell = cell_fn(rnn_size, use_peepholes=True, state_is_tuple=True)
    else:
      cell = cell_fn(rnn_size)

    if not infer and self.keep_prob < 1:
      cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    self.cell = cell = rnn_cell.MultiRNNCell([cell] * layer_depth, state_is_tuple=True)
    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
    self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])
    self.initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope('rnnlm'):
      with tf.device("/cpu:0"):
        self.embedding = tf.get_variable("embedding", [vocab_size, edim],
                                         initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(self.embedding, self.input_data))
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    def loop(prev, _):
      prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      return tf.nn.embedding_lookup(self.embedding, prev_symbol)

    with tf.variable_scope('output'):
      softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size],
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=True))
      softmax_b = tf.get_variable("softmax_b", [vocab_size])

      outputs, self.final_state = seq2seq.rnn_decoder(inputs, # [seq_length, batch_size, edim]
                                                      self.initial_state, cell,
                                                      loop_function=loop if infer else None,
                                                      scope='rnnlm')
      output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size]) # [seq_length * batch_size, edim]
      self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
      self.probs = tf.nn.softmax(self.logits)

    self.learning_rate = tf.Variable(0.0, trainable=False)
    self.loss = seq2seq.sequence_loss_by_example([self.logits],
                                              [tf.reshape(self.targets, [-1])],
                                              [tf.ones([batch_size * seq_length])],
                                              vocab_size)
    self.cost = (tf.reduce_sum(self.loss) / batch_size / seq_length)

    tvars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    self.perplexity = tf.exp(self.cost / seq_length)

    tf.scalar_summary("learning rate", self.learning_rate)
    tf.scalar_summary("cost", self.cost)
    tf.histogram_summary("loss", self.loss)
    tf.scalar_summary("perplexity", self.perplexity)
    self.merged = tf.merge_all_summaries()

  def sample(self, sess, chars, vocab, num=200, prime='The '):
    state = self.cell.zero_state(1, tf.float32).eval()
    prime = prime.decode('utf-8')

    for char in prime[:-1]:
      x = np.zeros((1, 1))
      x[0, 0] = vocab.get(char, 0)
      feed = {self.input_data: x, self.initial_state:state}
      [state] = sess.run([self.final_state], feed)

    def weighted_pick(weights):
      t = np.cumsum(weights)
      s = np.sum(weights)
      return(int(np.searchsorted(t, np.random.rand(1)*s)))

    ret = prime
    char = prime[-1]

    for _ in xrange(num):
      x = np.zeros((1, 1))
      x[0, 0] = vocab.get(char, 0)
      feed = {self.input_data: x, self.initial_state:state}
      [probs, state] = sess.run([self.probs, self.final_state], feed)
      p = probs[0]
      # sample = int(np.random.choice(len(p), p=p))
      sample = weighted_pick(p)
      pred = chars[sample]
      ret += pred
      char = pred

    return ret
