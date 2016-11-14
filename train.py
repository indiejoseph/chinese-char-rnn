# -*- coding: utf-8 -*-

import os
import time
import codecs
import numpy as np
import tensorflow as tf
import cPickle
import pprint
import string
import sys
from models.charrnn import CharRNN
from utils import TextLoader, normalize_unicodes

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("num_epochs", 25, "Epoch to train [25]")
flags.DEFINE_integer("rnn_size", 128, "The dimension of char embedding matrix [128]")
flags.DEFINE_integer("layer_depth", 2, "Number of layers for RNN")
flags.DEFINE_integer("batch_size", 50, "The size of batch [50]")
flags.DEFINE_integer("seq_length", 25, "The # of timesteps to unroll for [25]")
flags.DEFINE_float("learning_rate", .2, "Learning rate [.2]")
flags.DEFINE_float("decay_rate", 0.97, "Decay rate [0.97]")
flags.DEFINE_integer("num_sampled", 120, "Number of noise sampled [120]")
flags.DEFINE_float("keep_prob", 0.5, "Dropout rate")
flags.DEFINE_integer("save_every", 1000, "Save every")
flags.DEFINE_integer("valid_every", 500, "Validate every")
flags.DEFINE_float("grad_clip", 5., "clip gradients at this value")
flags.DEFINE_string("dataset_name", "news", "The name of datasets [news]")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("log_dir", "log", "Log directory [log]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample", "", "sample")
flags.DEFINE_boolean("export", False, "Export embedding")
FLAGS = flags.FLAGS

def gen_sample(sess, model, chars, vocab, num=200, prime='The ', sampling_type=1):
  state = sess.run(model.cell.zero_state(1, tf.float32))
  prime = prime.decode('utf-8')

  for char in prime[:-1]:
    x = np.zeros((1, 1))
    x[0, 0] = vocab[char]
    feed = {model.input_data: x}
    c, h = model.initial_state
    feed[c] = state[0]
    feed[h] = state[1]
    [state] = sess.run([model.final_state], feed)

  def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return int(np.searchsorted(t, np.random.rand(1)*s))

  ret = prime
  char = prime[-1]

  for _ in xrange(num):
    x = np.zeros((1, 1))
    x[0, 0] = vocab.get(char, 0)
    feed = {model.input_data: x, model.initial_state:state}
    [probs, state] = sess.run([model.probs, model.final_state], feed)
    p = probs[0]

    if sampling_type == 0:
        sample = np.argmax(p)
    elif sampling_type == 2:
        if char == ' ':
            sample = weighted_pick(p)
        else:
            sample = np.argmax(p)
    else: # sampling_type == 1 default:
        sample = weighted_pick(p)

    pred = chars[sample]
    ret += pred
    char = pred

  return ret


def compute_similarity(model, valid_size=16, valid_window=100, offset=0):
  # We pick a random validation set to sample nearest neighbors. Here we limit the
  # validation samples to the characters that have a low numeric ID, which by
  # construction are also the most frequent.
  # valid_size: Random set of characters to evaluate similarity on.
  # valid_size: Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(range(offset, offset + valid_window), valid_size, replace=False)
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(model.embedding), 1, keep_dims=True))
  normalized_embeddings = model.embedding / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  return similarity, valid_examples, valid_dataset

def run_epochs(sess, x, y, state, model, get_summary=True, is_training=True):
  start = time.time()
  feed = {model.input_data: x, model.targets: y}

  c, h = model.initial_state
  feed[c] = state[0]
  feed[h] = state[1]

  if is_training:
    extra_op = model.train_op
  else:
    extra_op = tf.no_op()

  if get_summary:
    fetchs = [model.merged_summary, model.final_state, model.cost, extra_op]
  else:
    fetchs = [model.final_state, model.cost, extra_op]

  res = sess.run(fetchs, feed)
  end = time.time()

  return res, end - start

def main(_):
  pp.pprint(FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    print " [*] Creating checkpoint directory..."
    os.makedirs(FLAGS.checkpoint_dir)

  data_loader = TextLoader(os.path.join(FLAGS.data_dir, FLAGS.dataset_name),
                           FLAGS.batch_size, FLAGS.seq_length)
  vocab_size = data_loader.vocab_size
  graph = tf.Graph()
  valid_size = 50
  valid_window = 100

  with tf.Session(graph=graph) as sess:
    graph_info = sess.graph

    with graph.as_default():
      with tf.name_scope('training'):
        train_model = CharRNN(vocab_size, FLAGS.batch_size,
                              FLAGS.layer_depth, FLAGS.rnn_size, FLAGS.num_sampled,
                              FLAGS.seq_length, FLAGS.grad_clip, FLAGS.keep_prob,
                              FLAGS.checkpoint_dir, FLAGS.dataset_name, infer=False)
      tf.get_variable_scope().reuse_variables()
      with tf.name_scope('validation'):
        valid_model = CharRNN(vocab_size, FLAGS.batch_size,
                              FLAGS.layer_depth, FLAGS.rnn_size, FLAGS.num_sampled,
                              FLAGS.seq_length, FLAGS.grad_clip, FLAGS.keep_prob,
                              FLAGS.checkpoint_dir, FLAGS.dataset_name, infer=True)
      with tf.name_scope('sample'):
        simple_model = CharRNN(vocab_size, 1,
                               FLAGS.layer_depth, FLAGS.rnn_size, FLAGS.num_sampled,
                               1, FLAGS.grad_clip, FLAGS.keep_prob,
                               FLAGS.checkpoint_dir, FLAGS.dataset_name, infer=True)

    train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/training', graph_info)
    valid_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/validate', graph_info)
    tf.initialize_all_variables().run()

    if FLAGS.sample:
      # load checkpoints
      if simple_model.load(sess, simple_model.checkpoint_dir, simple_model.dataset_name):
        print " [*] SUCCESS to load model for %s." % simple_model.dataset_name
      else:
        print " [!] Failed to load model for %s." % simple_model.dataset_name
        sys.exit(1)

      sample = normalize_unicodes(FLAGS.sample)
      print gen_sample(sess, simple_model, data_loader.chars, data_loader.vocab, 200, sample)

    elif FLAGS.export:
      print "Eval..."
      final_embeddings = train_model.embedding.eval(sess)
      emb_file = os.path.join(FLAGS.data_dir, FLAGS.dataset_name, 'emb.npy')
      print "Embedding shape: {}".format(final_embeddings.shape)
      np.save(emb_file, final_embeddings)

    else: # Train
      current_step = 0
      similarity, valid_examples, _ = compute_similarity(train_model, valid_size, valid_window, 6)

      # run it!
      for e in xrange(FLAGS.num_epochs):
        sess.run(tf.assign(train_model.learning_rate, FLAGS.learning_rate * (FLAGS.decay_rate ** e)))

        data_loader.reset_batch_pointer()

        state = sess.run(train_model.initial_state)
        train_iter_size = FLAGS.batch_size * FLAGS.seq_length
        valid_iter_size = data_loader.num_valid_batches * FLAGS.seq_length
        train_iters = 0
        valid_iters = 0
        train_costs = 0
        valid_costs = 0

        # iterate by batch
        for b in xrange(data_loader.num_batches):
          x, y = data_loader.next_batch()
          res, time_batch = run_epochs(sess, x, y, state, train_model)
          summary = res[0]
          train_cost = res[2]
          state = res[1]
          train_iters += train_iter_size
          train_costs += train_cost
          train_perplexity = np.exp(train_costs / train_iters)

          if current_step % FLAGS.valid_every == 0:
            valid_state = sess.run(valid_model.initial_state)
            valid_cost = 0

            for vb in xrange(data_loader.num_valid_batches):
              res, valid_time_batch = run_epochs(sess, data_loader.x_valid[vb], data_loader.y_valid[vb],
                                                 valid_state, valid_model, False)
              valid_cost += res[1]
              valid_iters += valid_iter_size
              valid_costs += valid_cost
              valid_perplexity = np.exp(valid_costs / valid_iters)

            valid_writer.add_summary(tf.scalar_summary("valid_perplexity", valid_perplexity).eval(), current_step)
            valid_writer.flush()

            print "### valid_perplexity = {:.2f}, time/batch = {:.2f}" \
              .format(valid_perplexity, valid_time_batch)

            # write summary
            train_writer.add_summary(summary, current_step)
            train_writer.flush()

            # Write a similarity log
            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            log_str = ""
            sim = similarity.eval()
            for i in xrange(valid_size):
              valid_word = data_loader.chars[valid_examples[i]]
              top_k = 8 # number of nearest neighbors
              nearest = (-sim[i, :]).argsort()[1:top_k+1]
              log_str = log_str + "Nearest to %s:" % valid_word
              for k in xrange(top_k):
                close_word = data_loader.chars[nearest[k]]
                log_str = "%s %s," % (log_str, close_word)
              log_str = log_str + "\n"
            print log_str
            # Write to log
            text_file = codecs.open(FLAGS.log_dir + "/similarity.txt", "w", "utf-8")
            text_file.write(log_str)
            text_file.close()

          # print log
          print "{}/{} (epoch {}) train_perplexity = {:.2f} last_valid = {:.2f} time/batch = {:.2f}" \
              .format(e * data_loader.num_batches + b,
                      FLAGS.num_epochs * data_loader.num_batches,
                      e, train_perplexity, valid_perplexity, time_batch)

          # save model to checkpoint
          if (e * data_loader.num_batches + b) % FLAGS.save_every == 0:
            train_model.save(sess, FLAGS.checkpoint_dir, FLAGS.dataset_name)
            print "model saved to {}".format(FLAGS.checkpoint_dir)

          current_step = tf.train.global_step(sess, train_model.global_step)


if __name__ == '__main__':
  tf.app.run()
