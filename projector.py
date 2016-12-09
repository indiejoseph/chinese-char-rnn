from models.charrnn import CharRNN
import tensorflow as tf
import cPickle
import os
import codecs
from tensorflow.contrib.tensorboard.plugins import projector

checkpoint_dir = 'checkpoint/news'
vocab_file = 'data/news/vocab.pkl'

graph = tf.Graph()

with open(vocab_file, 'rb') as f:
  chars = cPickle.load(f)
  chars[4] = '\\s'
  chars[5] = '\\n'
  f.close()

with codecs.open('log/metadata.tsv', 'w', 'utf-8') as f:
  f.write('\n'.join(chars))
  f.close()

vocab_size = len(chars)
rnn_size = 128

def load_embedding():
  embedding = tf.get_variable("training/embedding",
    initializer=tf.random_uniform([vocab_size, rnn_size], -1.0, 1.0))

  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

  with tf.Session() as sess:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    embedding_value = sess.run(embedding)

  return embedding_value


embedding_value = load_embedding()
embedding_var = tf.Variable(embedding_value, "embedding_var")

# Use the same LOG_DIR where you stored your checkpoint.
sw = tf.train.SummaryWriter('log')
projector_config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = projector_config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = 'log/metadata.tsv'

saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.save(sess, os.path.join("log", "embedding_var.ckpt"))

projector.visualize_embeddings(sw, projector_config)
