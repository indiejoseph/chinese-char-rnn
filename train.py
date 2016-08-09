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
flags.DEFINE_integer("rnn_size", 256, "The dimension of char embedding matrix [256]")
flags.DEFINE_integer("layer_depth", 2, "Number of layers for RNN")
flags.DEFINE_integer("batch_size", 50, "The size of batch [50]")
flags.DEFINE_integer("seq_length", 25, "The # of timesteps to unroll for [25]")
flags.DEFINE_float("learning_rate", .002, "Learning rate [.002]")
flags.DEFINE_string("model", "lstm", "RNN model [lstm]")
flags.DEFINE_float("decay_rate", 0.9, "Decay rate [0.9]")
flags.DEFINE_integer("nce_samples", 25, "NCE sample size [25]")
flags.DEFINE_float("keep_prob", 0.5, "Dropout rate")
flags.DEFINE_integer("save_every", 1000, "Save every")
flags.DEFINE_integer("valid_every", 1000, "Validate every")
flags.DEFINE_integer("summary_every", 1000, "Write summary every")
flags.DEFINE_boolean("use_peepholes", True, "use peepholes")
flags.DEFINE_float("grad_clip", 5., "clip gradients at this value")
flags.DEFINE_string("dataset_name", "news", "The name of datasets [news]")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("log_dir", "log", "Log directory [log]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample", "", "sample")
flags.DEFINE_boolean("export", False, "Export embedding")
FLAGS = flags.FLAGS

def compute_similarity (model, valid_size=16, valid_window=100):
  # We pick a random validation set to sample nearest neighbors. Here we limit the
  # validation samples to the characters that have a low numeric ID, which by
  # construction are also the most frequent.
  # valid_size: Random set of characters to evaluate similarity on.
  # valid_size: Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(model.embedding), 1, keep_dims=True))
  normalized_embeddings = model.embedding / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  return similarity, valid_examples, valid_dataset


def main(_):
  pp.pprint(FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    print " [*] Creating checkpoint directory..."
    os.makedirs(FLAGS.checkpoint_dir)

  if FLAGS.sample:
    infer = True
  else:
    infer = False

  data_loader = TextLoader(os.path.join(FLAGS.data_dir, FLAGS.dataset_name),
                           FLAGS.batch_size, FLAGS.seq_length)
  vocab_size = data_loader.vocab_size
  graph = tf.Graph()
  valid_size = 25
  valid_window = 100

  with tf.Session(graph=graph) as sess:
    graph_info = sess.graph
    model = CharRNN(sess, vocab_size, FLAGS.batch_size,
                    FLAGS.layer_depth, FLAGS.rnn_size, FLAGS.nce_samples, FLAGS.model,
                    FLAGS.use_peepholes, FLAGS.seq_length, FLAGS.grad_clip, FLAGS.keep_prob,
                    FLAGS.checkpoint_dir, FLAGS.dataset_name, infer=infer)
    writer = tf.train.SummaryWriter(FLAGS.log_dir, graph_info)
    tf.initialize_all_variables().run()

    if FLAGS.sample:
      # load checkpoints
      if model.load(model.checkpoint_dir, model.dataset_name):
        print " [*] SUCCESS to load model for %s." % model.dataset_name
      else:
        print " [!] Failed to load model for %s." % model.dataset_name
        sys.exit(1)

      sample = normalize_unicodes(FLAGS.sample)
      print model.sample(sess, data_loader.chars, data_loader.vocab, 200, sample)

    elif FLAGS.export:
      print "Eval..."
      final_embeddings = model.embedding.eval(sess)
      emb_file = os.path.join(FLAGS.data_dir, FLAGS.dataset_name, 'emb.npy')
      print "Embedding shape: {}".format(final_embeddings.shape)
      np.save(emb_file, final_embeddings)

    else: # Train
      current_step = 0
      similarity, valid_examples, _ = compute_similarity(model, valid_size, valid_window)

      # run it!
      for e in xrange(FLAGS.num_epochs):
        sess.run(tf.assign(model.learning_rate, FLAGS.learning_rate * (FLAGS.decay_rate ** e)))

        data_loader.reset_batch_pointer()

        # assign final state to rnn
        state_list = []
        for c, h in model.initial_state:
          state_list.extend([c.eval(), h.eval()])

        # iterate by batch
        for b in xrange(data_loader.num_batches):
          start = time.time()
          x, y = data_loader.next_batch()
          feed = {model.input_data: x, model.targets: y}

          for i in range(len(model.initial_state)):
            c, h = model.initial_state[i]
            feed[c], feed[h] = state_list[i*2:(i+1)*2]

          fetchs = [model.merged, model.cost, model.train_op]
          for c, h in model.final_state:
            fetchs.extend([c, h])

          res = sess.run(fetchs, feed)
          summary = res[0]
          train_cost = res[1]
          state_list = res[3:]
          end = time.time()

          if current_step % FLAGS.summary_every == 0:
            writer.add_summary(summary, current_step)

          if current_step % FLAGS.valid_every == 0:
            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            log_str = ""
            sim = similarity.eval()
            for i in xrange(valid_size):
              valid_word = data_loader.chars[valid_examples[i]]
              top_k = 10 # number of nearest neighbors
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

          print "{}/{} (epoch {}), train_loss = {:.2f}, time/batch = {:.2f}" \
              .format(e * data_loader.num_batches + b,
                      FLAGS.num_epochs * data_loader.num_batches,
                      e, train_cost, end - start)

          if (e * data_loader.num_batches + b) % FLAGS.save_every == 0:
            # save to checkpoint
            model.save(FLAGS.checkpoint_dir, FLAGS.dataset_name)
            print "model saved to {}".format(FLAGS.checkpoint_dir)

          current_step = tf.train.global_step(sess, model.global_step)


if __name__ == '__main__':
  tf.app.run()
