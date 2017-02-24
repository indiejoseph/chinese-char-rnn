# -*- coding: utf-8 -*-

import os
import time
import codecs
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import cPickle
import pprint
import string
import sys
from models.charrnn import CharRNN
from utils import TextLoader

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("num_epochs", 25, "Epoch to train [25]")
flags.DEFINE_integer("rnn_size", 1000, "The dimension of char embedding matrix [1000]")
flags.DEFINE_integer("num_units", 100, "The dimension of char embedding matrix [100]")
flags.DEFINE_integer("layer_depth", 2, "Number of layers for RNN [2]")
flags.DEFINE_integer("batch_size", 120, "The size of batch [120]")
flags.DEFINE_integer("seq_length", 20, "The # of timesteps to unroll for [20]")
flags.DEFINE_float("learning_rate", 1, "Learning rate [1]")
flags.DEFINE_float("decay_rate", 0.9, "Decay rate for SDG")
flags.DEFINE_float("keep_prob", 0.5, "Dropout rate [0.5]")
flags.DEFINE_float("grad_clip", 2.0, "Grad clip [2.0]")
flags.DEFINE_integer("valid_every", 1000, "Validate every")
flags.DEFINE_integer("num_sampled", 5, "Number of softmax sample")
flags.DEFINE_string("dataset_name", "news", "The name of datasets [news]")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("log_dir", "log", "Log directory [log]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("export", False, "Export embedding")
FLAGS = flags.FLAGS


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

def run_epochs(sess, x, y, state, model, is_training=True):
  start = time.time()
  feed = {model.input_data: x, model.targets: y}

  if state is not None:
    feed[model.initial_state] = state

  if is_training:
    extra_op = model.train_op
  else:
    extra_op = tf.no_op()

  fetchs = {
    "final_state": model.final_state,
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
  valid_size = 50
  valid_window = 100

  with tf.name_scope('training'):
    train_model = CharRNN(vocab_size, FLAGS.batch_size,
                          FLAGS.layer_depth, FLAGS.rnn_size, FLAGS.num_units,
                          FLAGS.seq_length, FLAGS.keep_prob,
                          FLAGS.grad_clip, FLAGS.num_sampled,
                          is_training=True)

  tf.get_variable_scope().reuse_variables()

  with tf.name_scope('validation'):
    valid_model = CharRNN(vocab_size, FLAGS.batch_size,
                          FLAGS.layer_depth, FLAGS.rnn_size, FLAGS.num_units,
                          FLAGS.seq_length, FLAGS.keep_prob,
                          FLAGS.grad_clip, FLAGS.num_sampled,
                          is_training=False)

  with tf.Session() as sess:
    tf.global_variables_initializer().run()

    if FLAGS.export:
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
        data_loader.reset_batch_pointer()

        train_iters = 0
        valid_iters = 0
        train_costs = 0
        valid_costs = 0
        state = None

        # decay learning rate
        sess.run(tf.assign(train_model.lr, FLAGS.learning_rate * (FLAGS.decay_rate ** e)))

        # iterate by batch
        for b in xrange(data_loader.num_batches):
          x, y = data_loader.next_batch()
          res, time_batch = run_epochs(sess, x, y, state, train_model)
          train_cost = res["cost"]
          state = res["final_state"]
          train_iters += 1
          train_costs += train_cost
          train_perplexity = np.exp(train_costs / train_iters)

          if current_step % FLAGS.valid_every == 0:
            valid_state = None
            valid_cost = 0

            for vb in xrange(data_loader.num_valid_batches):
              res, valid_time_batch = run_epochs(sess, data_loader.x_valid[vb], data_loader.y_valid[vb],
                                                 valid_state, valid_model, False)
              valid_state = res["final_state"]
              valid_iters += 1
              valid_cost += res["cost"]
              valid_costs += res["cost"]
              valid_perplexity = np.exp(valid_costs / valid_iters)

            print "### valid_perplexity = {:.2f}, time/batch = {:.2f}" \
              .format(valid_perplexity, valid_time_batch)

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
          print "{}/{} (epoch {}) cost = {:.2f}({:.2f}) train = {:.2f}({:.2f}) time/batch = {:.2f} chars/sec = {:.2f}k"\
              .format(e * data_loader.num_batches + b,
                      FLAGS.num_epochs * data_loader.num_batches,
                      e, train_cost, (valid_cost / data_loader.num_valid_batches), train_perplexity, valid_perplexity,
                      time_batch, (FLAGS.batch_size * FLAGS.seq_length) / time_batch / 1000)

          current_step = tf.train.global_step(sess, train_model.global_step)

        train_model.save(sess, FLAGS.checkpoint_dir, FLAGS.dataset_name)
        print "model saved to {}".format(FLAGS.checkpoint_dir)


if __name__ == '__main__':
  tf.app.run()
