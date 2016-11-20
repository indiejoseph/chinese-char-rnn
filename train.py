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
from models.bytenet import ByteNet
from utils import TextLoader, normalize_unicodes

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("num_epochs", 25, "Epoch to train [25]")
flags.DEFINE_integer("residual_channels", 256, "Residual channels")
flags.DEFINE_string("dialations", "1,2,4,8,16,1,2,4,8,16,1,2,4,8,16,1,2,4,8,16,1,2,4,8,16", "Dialations")
flags.DEFINE_integer("filter_width", 3, "Filter width for conv")
flags.DEFINE_integer("batch_size", 50, "The size of batch [50]")
flags.DEFINE_integer("seq_length", 25, "The # of timesteps to unroll for [25]")
flags.DEFINE_float("learning_rate", 1, "Learning rate [1]")
flags.DEFINE_float("decay_rate", 0.97, "Decay rate [0.97]")
flags.DEFINE_integer("test_every", 1000, "Validate every")
flags.DEFINE_float("grad_clip", 5., "clip gradients at this value")
flags.DEFINE_string("dataset_name", "news", "The name of datasets [news]")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("log_dir", "log", "Log directory [log]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample", "", "sample")
flags.DEFINE_boolean("export", False, "Export embedding")
FLAGS = flags.FLAGS

def gen_sample(sess, model, chars, vocab, num=200, prime='The ', sampling_type=1):
  prime = prime.decode('utf-8')

  for char in prime[:-1]:
    x = np.zeros((1, 1))
    x[0, 0] = vocab[char]
    feed = {model.sentence: x}
    _ = sess.run([], feed)

  def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return int(np.searchsorted(t, np.random.rand(1)*s))

  ret = prime
  char = prime[-1]

  for _ in xrange(num):
    x = np.zeros((1, 1))
    x[0, 0] = vocab.get(char, 0)
    feed = {model.sentence: x}
    [probs] = sess.run([model.probs], feed)
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


def compute_similarity(model, test_size=16, test_window=100, offset=0):
  # We pick a random validation set to sample nearest neighbors. Here we limit the
  # validation samples to the characters that have a low numeric ID, which by
  # construction are also the most frequent.
  # test_size: Random set of characters to evaluate similarity on.
  # test_size: Only pick dev samples in the head of the distribution.
  test_examples = np.random.choice(range(offset, offset + test_window), test_size, replace=False)
  test_dataset = tf.constant(test_examples, dtype=tf.int32)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(model.w_source_embedding), 1, keep_dims=True))
  normalized_embeddings = model.w_source_embedding / norm
  test_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            test_dataset)
  similarity = tf.matmul(test_embeddings, normalized_embeddings, transpose_b=True)

  return similarity, test_examples, test_dataset

def run_epochs(sess, x, y, model, is_training=True):
  start = time.time()
  feed = {model.sentence: x, model.targets: y}

  if is_training:
    extra_op = model.train_op
  else:
    extra_op = tf.no_op()

  fetchs = {
    "cost": model.cost,
    "extra_op": extra_op
  }

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
  test_size = 50
  test_window = 100

  with tf.Session(graph=graph) as sess:
    graph_info = sess.graph

    with graph.as_default():
      with tf.name_scope('training'):
        train_model = ByteNet(vocab_size, vocab_size, FLAGS.residual_channels, FLAGS.batch_size,
                              FLAGS.seq_length, FLAGS.filter_width, FLAGS.filter_width,
                              FLAGS.dialations, FLAGS.dialations,
                              FLAGS.grad_clip,
                              checkpoint_dir=FLAGS.checkpoint_dir, dataset_name=FLAGS.dataset_name)
      tf.get_variable_scope().reuse_variables()
      with tf.name_scope('validation'):
        test_model = ByteNet(vocab_size, vocab_size, FLAGS.residual_channels, FLAGS.batch_size,
                              FLAGS.seq_length, FLAGS.filter_width, FLAGS.filter_width,
                              FLAGS.dialations, FLAGS.dialations,
                              FLAGS.grad_clip,
                              checkpoint_dir=FLAGS.checkpoint_dir, dataset_name=FLAGS.dataset_name)
      with tf.name_scope('sample'):
        simple_model = ByteNet(vocab_size, vocab_size, FLAGS.residual_channels, 1,
                               FLAGS.seq_length, FLAGS.filter_width, FLAGS.filter_width,
                               FLAGS.dialations, FLAGS.dialations,
                               FLAGS.grad_clip,
                               checkpoint_dir=FLAGS.checkpoint_dir, dataset_name=FLAGS.dataset_name)

    train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/training', graph_info)
    test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/validate', graph_info)
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
      final_embeddings = train_model.w_source_embedding.eval(sess)
      emb_file = os.path.join(FLAGS.data_dir, FLAGS.dataset_name, 'emb.npy')
      print "Embedding shape: {}".format(final_embeddings.shape)
      np.save(emb_file, final_embeddings)

    else: # Train
      current_step = 0
      similarity, test_examples, _ = compute_similarity(train_model, test_size, test_window, 6)

      # run it!
      for e in xrange(FLAGS.num_epochs):
        sess.run(tf.assign(train_model.learning_rate, FLAGS.learning_rate * (FLAGS.decay_rate ** e)))

        data_loader.reset_batch_pointer()

        train_iters = 0
        test_iters = 0
        train_costs = 0
        test_costs = 0

        # iterate by batch
        for b in xrange(data_loader.num_batches):
          x, y = data_loader.next_batch()
          res, time_batch = run_epochs(sess, x, y, train_model)
          train_cost = res["cost"]
          train_iters += 1
          train_costs += train_cost
          train_perplexity = np.exp(train_costs / train_iters)

          if current_step % FLAGS.test_every == 0:
            for vb in xrange(data_loader.num_test_batches):
              res, test_time_batch = run_epochs(sess, data_loader.x_test[vb], data_loader.y_test[vb],
                                                 test_model, False)
              test_iters += 1
              test_costs += res["cost"]
              test_perplexity = np.exp(test_costs / test_iters)

            test_writer.add_summary(tf.scalar_summary("test_perplexity", test_perplexity).eval(), current_step)
            test_writer.flush()

            print "### test_perplexity = {:.2f}, time/batch = {:.2f}" \
              .format(test_perplexity, test_time_batch)

            # write summary
            train_writer.add_summary(tf.scalar_summary("train_perplexity", train_perplexity).eval(), current_step)
            train_writer.flush()

            # Write a similarity log
            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            log_str = ""
            sim = similarity.eval()
            for i in xrange(test_size):
              test_word = data_loader.chars[test_examples[i]]
              top_k = 8 # number of nearest neighbors
              nearest = (-sim[i, :]).argsort()[1:top_k+1]
              log_str = log_str + "Nearest to %s:" % test_word
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
          print "{}/{} (epoch {}) cost = {:.2f} train = {:.2f} test = {:.2f} time/batch = {:.2f}" \
              .format(e * data_loader.num_batches + b,
                      FLAGS.num_epochs * data_loader.num_batches,
                      e, train_cost, train_perplexity, test_perplexity, time_batch)

          current_step = tf.train.global_step(sess, train_model.global_step)

        train_model.save(sess, FLAGS.checkpoint_dir, FLAGS.dataset_name)
        print "model saved to {}".format(FLAGS.checkpoint_dir)


if __name__ == '__main__':
  tf.app.run()
