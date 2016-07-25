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
flags.DEFINE_integer("edim", 128, "The dimension of char embedding matrix [128]")
flags.DEFINE_integer("rnn_size", 1024, "The size of state for RNN")
flags.DEFINE_integer("layer_depth", 2, "Number of layers for RNN")
flags.DEFINE_integer("batch_size", 50, "The size of batch [50]")
flags.DEFINE_integer("seq_length", 25, "The # of timesteps to unroll for [25]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate [0.001]")
flags.DEFINE_float("decay_rate", 0.95, "Decay of SGD [0.95]")
flags.DEFINE_float("keep_prob", 0.5, "Dropout rate")
flags.DEFINE_float("l2_reg_lambda", 1e-4, "L2 Normalization")
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
                    FLAGS.rnn_size, FLAGS.layer_depth, FLAGS.edim, FLAGS.l2_reg_lambda,
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
      # assign learning rate to model
      sess.run(tf.assign(model.learning_rate, FLAGS.learning_rate))
      learning_rate = FLAGS.learning_rate
      step = 0

      for e in xrange(FLAGS.num_epochs):
        data_loader.reset_batch_pointer()

        for b in xrange(data_loader.num_batches):
          start = time.time()
          x, y = data_loader.next_batch()
          feed = {model.input_data: x, model.targets: y}

          summary, perplexity, train_cost, _ = sess.run(
            [model.merged, model.perplexity,
             model.cost,
             model.train_op], feed)
          end = time.time()

          if step % FLAGS.summary_every == 0:
            writer.add_summary(summary, step)

          step += 1

          print "{}/{} (epoch {}), train_loss = {:.2f}, perplexity = {:.2f}, time/batch = {:.2f}, lr = {:.4f}" \
              .format(e * data_loader.num_batches + b,
                      FLAGS.num_epochs * data_loader.num_batches,
                      e, train_cost, perplexity, end - start, learning_rate)

          if (e * data_loader.num_batches + b) % FLAGS.save_every == 0:
            # save to checkpoint
            model.save(FLAGS.checkpoint_dir, FLAGS.dataset_name)
            print "model saved to {}".format(FLAGS.checkpoint_dir)


if __name__ == '__main__':
  tf.app.run()
