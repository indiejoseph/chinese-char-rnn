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
flags.DEFINE_integer("edim", 150, "The dimension of char embedding matrix [150]")
flags.DEFINE_integer("ldim", 50, "The dimension of language embedding matrix [50]")
flags.DEFINE_integer("rnn_size", 600, "The size of state for RNN [600]")
flags.DEFINE_integer("layer_depth", 2, "Number of layers for RNN [2]")
flags.DEFINE_integer("batch_size", 30, "The size of batch [30]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate [1e-4]")
flags.DEFINE_float("keep_prob", 0.5, "Dropout rate")
flags.DEFINE_integer("save_every", 1000, "Save every")
flags.DEFINE_string("model", "gru", "rnn, lstm or gru")
flags.DEFINE_float("grad_clip", 5., "clip gradients at this value")
flags.DEFINE_string("dataset_name", "news", "The name of datasets [news]")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample", "", "sample")
flags.DEFINE_integer("lang", 0, "language")
flags.DEFINE_boolean("export", False, "Export embedding")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    print(" [*] Creating checkpoint directory...")
    os.makedirs(FLAGS.checkpoint_dir)

  if FLAGS.sample:
    infer = True
  else:
    infer = False

  data_loader = TextLoader(os.path.join(FLAGS.data_dir, FLAGS.dataset_name), FLAGS.batch_size)
  vocab_size = data_loader.vocab_size
  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  with tf.Session(config=config) as sess:
    model = CharRNN(sess, vocab_size, FLAGS.learning_rate, FLAGS.batch_size,
                    FLAGS.rnn_size, FLAGS.layer_depth, FLAGS.edim, FLAGS.ldim, data_loader.lang_size,
                    FLAGS.model, data_loader.seq_length, FLAGS.grad_clip, FLAGS.keep_prob,
                    FLAGS.checkpoint_dir, infer=infer)

    tf.initialize_all_variables().run()
    writer = tf.train.SummaryWriter("log", graph=sess.graph)

    if FLAGS.sample:
      # load checkpoints
      if model.load(model.checkpoint_dir, FLAGS.dataset_name):
        print(" [*] SUCCESS to load model for %s." % FLAGS.dataset_name)
      else:
        print(" [!] Failed to load model for %s." % FLAGS.dataset_name)
        sys.exit(1)

      sample = normalizeUnicodes(FLAGS.sample)
      print model.sample(sess, data_loader.chars, data_loader.vocab, 200, sample, FLAGS.lang) # 0 = eng

    elif FLAGS.export:
      print("Eval...")
      final_embeddings = model.embedding.eval(sess)
      emb_file = os.path.join(FLAGS.data_dir, FLAGS.dataset_name, 'emb.npy')
      print "Embedding shape: {}".format(final_embeddings.shape)
      np.save(emb_file, final_embeddings)

    else: # Train
      for e in xrange(FLAGS.num_epochs):
        data_loader.reset_batch_pointer()
        state = model.initial_state.eval()
        for b in xrange(data_loader.num_batches):
          start = time.time()
          x, y, z = data_loader.next_batch()
          feed = {model.input_data: x, model.targets: y, model.langs: z, model.initial_state: state}
          train_cost, state, step, summary, _ = sess.run([model.cost, model.final_state, model.global_step, model.summary, model.train_op], feed)
          end = time.time()

          writer.add_summary(summary, step)

          print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
              .format(e * data_loader.num_batches + b,
                      FLAGS.num_epochs * data_loader.num_batches,
                      e, train_cost, end - start)

          if (e * data_loader.num_batches + b) % FLAGS.save_every == 0:
            model.save(FLAGS.checkpoint_dir, FLAGS.dataset_name)
            print "model saved to {}".format(FLAGS.checkpoint_dir)


if __name__ == '__main__':
  tf.app.run()
