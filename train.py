# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import tensorflow as tf
import cPickle
import pprint
import string
import sys
from models.charrnn import CharRNN
from utils import TextLoader, normalizeUnicodes

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("num_epochs", 25, "Epoch to train [25]")
flags.DEFINE_integer("edim", 256, "The dimension of char embedding matrix [256]")
flags.DEFINE_integer("rnn_size", 1024, "The size of state for RNN")
flags.DEFINE_integer("layer_depth", 2, "Number of layers for RNN")
flags.DEFINE_integer("batch_size", 50, "The size of batch [50]")
flags.DEFINE_integer("seq_length", 25, "The # of timesteps to unroll for [25]")
flags.DEFINE_float("learning_rate", 1.0, "Learning rate [1.0]")
flags.DEFINE_integer("nce_samples", 10, "NCE sample size [10]")
flags.DEFINE_float("keep_prob", 0.5, "Dropout rate")
flags.DEFINE_integer("save_every", 1000, "Save every")
flags.DEFINE_integer("summary_every", 100, "Write summary every")
flags.DEFINE_string("model", "gru", "rnn, lstm or gru")
flags.DEFINE_boolean("use_peepholes", True, "use peepholes")
flags.DEFINE_float("grad_clip", 5., "clip gradients at this value")
flags.DEFINE_string("dataset_name", "news", "The name of datasets [news]")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("log_dir", "log", "Log directory [log]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample", "", "sample")
flags.DEFINE_boolean("export", False, "Export embedding")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    print(" [*] Creating checkpoint directory...")
    os.makedirs(FLAGS.checkpoint_dir)

  if FLAGS.sample:
    infer = True
  else:
    infer = False

  data_loader = TextLoader(os.path.join(FLAGS.data_dir, FLAGS.dataset_name),
                           FLAGS.batch_size, FLAGS.seq_length)
  vocab_size = data_loader.vocab_size
  graph = tf.Graph()

  with tf.Session(graph=graph) as sess:
    graph_info = sess.graph
    model = CharRNN(sess, vocab_size, FLAGS.batch_size,
                    FLAGS.rnn_size, FLAGS.layer_depth, FLAGS.edim, FLAGS.nce_samples,
                    FLAGS.model, FLAGS.use_peepholes, FLAGS.seq_length, FLAGS.grad_clip, FLAGS.keep_prob,
                    FLAGS.checkpoint_dir, FLAGS.dataset_name, infer=infer)
    writer = tf.train.SummaryWriter(FLAGS.log_dir, graph_info)
    tf.initialize_all_variables().run()

    if FLAGS.sample:
      # load checkpoints
      if model.load(model.checkpoint_dir, model.dataset_name):
        print(" [*] SUCCESS to load model for %s." % model.dataset_name)
      else:
        print(" [!] Failed to load model for %s." % model.dataset_name)
        sys.exit(1)

      sample = normalizeUnicodes(FLAGS.sample)
      print model.sample(sess, data_loader.chars, data_loader.vocab, 200, sample)

    elif FLAGS.export:
      print("Eval...")
      final_embeddings = model.embedding.eval(sess)
      emb_file = os.path.join(FLAGS.data_dir, FLAGS.dataset_name, 'emb.npy')
      print "Embedding shape: {}".format(final_embeddings.shape)
      np.save(emb_file, final_embeddings)

    else: # Train
      sess.run(tf.assign(model.learning_rate, FLAGS.learning_rate))

      # run it!
      for e in xrange(FLAGS.num_epochs):
        data_loader.reset_batch_pointer()

        # assign initial state to rnn
        state = []
        for i, s in enumerate(model.initial_state):
          state.append(s.eval())

        # iterate by batch
        for b in xrange(data_loader.num_batches):
          start = time.time()
          x, y = data_loader.next_batch()
          feed = {model.input_data: x, model.targets: y}

          # assign final state to rnn
          for i, state in enumerate(state):
            feed[model.initial_state[i]] = state

          fetchs = [model.merged, model.cost, model.train_op] + list(model.final_state)

          res = sess.run(fetchs, feed)
          current_step = tf.train.global_step(sess, model.global_step)
          summary = res[0]
          train_cost = res[1]
          state = res[3:]
          end = time.time()

          if current_step % FLAGS.summary_every == 0:
            writer.add_summary(summary, current_step)

          print "{}/{} (epoch {}), train_loss = {:.2f}, time/batch = {:.2f}" \
              .format(e * data_loader.num_batches + b,
                      FLAGS.num_epochs * data_loader.num_batches,
                      e, train_cost, end - start)

          if (e * data_loader.num_batches + b) % FLAGS.save_every == 0:
            # save to checkpoint
            model.save(FLAGS.checkpoint_dir, FLAGS.dataset_name)
            print "model saved to {}".format(FLAGS.checkpoint_dir)


if __name__ == '__main__':
  tf.app.run()
