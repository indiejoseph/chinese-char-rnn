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
flags.DEFINE_integer("edim", 200, "The dimension of char embedding matrix [200]")
flags.DEFINE_integer("ldim", 100, "The dimension of language embedding matrix [100]")
flags.DEFINE_integer("rnn_size", 900, "The size of state for RNN [900]")
flags.DEFINE_integer("layer_depth", 2, "Number of layers for RNN [2]")
flags.DEFINE_integer("batch_size", 30, "The size of batch [30]")
flags.DEFINE_float("learning_rate", 2e-3, "Learning rate [2e-3]")
flags.DEFINE_float("decay_rate", 0.97, "decay rate for optimizer [0.97]")
flags.DEFINE_float("keep_prob", .5, "Dropout rate")
flags.DEFINE_integer("save_every", 1000, "Save every")
flags.DEFINE_string("model", "lstm", "rnn, lstm or gru")
flags.DEFINE_float("grad_clip", 5., "clip gradients at this value")
flags.DEFINE_string("dataset_name", "news", "The name of datasets [news]")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_integer("valid_every", 1000, "Do validation every")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample", "", "sample")
flags.DEFINE_integer("lang", 0, "language") # 0 = eng
flags.DEFINE_boolean("export", False, "Export embedding")
flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
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
  config=tf.ConfigProto()
  config.gpu_options.allocator_type = "BFC"
  config.allow_soft_placement = True
  config.log_device_placement = FLAGS.log_device_placement
  sess = tf.Session(config=config)

  model = CharRNN(sess, vocab_size, FLAGS.learning_rate, FLAGS.batch_size,
                  FLAGS.rnn_size, FLAGS.layer_depth, FLAGS.edim, FLAGS.ldim, data_loader.lang_size,
                  FLAGS.model, data_loader.seq_length, FLAGS.grad_clip, FLAGS.keep_prob,
                  FLAGS.checkpoint_dir, infer=infer)

  init = tf.initialize_all_variables()
  sess.run(init)
  tf.train.start_queue_runners(sess=sess)

  writer = tf.train.SummaryWriter("log", sess.graph)
  tf.train.write_graph(sess.graph, 'log', 'graph.pb', as_text=True)

  if FLAGS.sample:
    # load checkpoints
    if model.load(model.checkpoint_dir, FLAGS.dataset_name):
      print(" [*] SUCCESS to load model for %s." % FLAGS.dataset_name)
    else:
      print(" [!] Failed to load model for %s." % FLAGS.dataset_name)
      sys.exit(1)

    sample = normalizeUnicodes(FLAGS.sample)
    print model.sample(sess, data_loader.chars, data_loader.vocab, 200, sample, FLAGS.lang)

  elif FLAGS.export:
    print("Eval...")
    final_embeddings = model.embedding.eval(session=sess)
    emb_file = os.path.join(FLAGS.data_dir, FLAGS.dataset_name, 'emb.npy')
    print "Embedding shape: {}".format(final_embeddings.shape)
    np.save(emb_file, final_embeddings)

  else: # Train
    x_valid, y_valid, z_valid = data_loader.get_valid()
    valid_cost = 0

    for e in xrange(FLAGS.num_epochs):
      sess.run(tf.assign(model.learning_rate, FLAGS.learning_rate * (FLAGS.decay_rate ** e)))
      data_loader.reset_batch_pointer()
      for b in xrange(data_loader.num_batches):
        start = time.time()
        x, y, z = data_loader.next_batch()
        state = model.initial_state.eval(session=sess) # fresh state for each sentence
        feed = {model.input_data: x, model.targets: y, model.langs: z, model.initial_state: state}
        train_cost, step, summary, _ = sess.run([model.cost, model.global_step, model.summary, model.train_op], feed)
        end = time.time()
        step = (e * data_loader.num_batches + b)

        writer.add_summary(summary, step)

        print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}, valid = {:.3f}" \
            .format(e * data_loader.num_batches + b,
                    FLAGS.num_epochs * data_loader.num_batches,
                    e, train_cost, end - start, valid_cost)

        if step % FLAGS.save_every == 0:
          model.save(FLAGS.checkpoint_dir, FLAGS.dataset_name)
          print "model saved to {}".format(FLAGS.checkpoint_dir)

        if step % FLAGS.valid_every == 0:
          state = model.initial_state.eval(session=sess) # fresh state for each sentence
          feed = {model.input_data: x_valid, model.targets: y_valid, model.langs: z_valid, model.initial_state: state}
          output = sess.run([model.cost], feed)
          valid_cost = output[0]

if __name__ == '__main__':
  tf.app.run()
