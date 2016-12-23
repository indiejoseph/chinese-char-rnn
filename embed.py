import os
import tensorflow as tf

def get_embedding(ckp_dir, sess):
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(ckp_dir)
  ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
  saver.restore(sess, os.path.join(ckp_dir, ckpt_name))

vocab_size = 3076
rnn_size = 128

if __name__ == "__main__":
  with tf.Session() as sess:
    embedding = tf.get_variable("embedding",
      initializer=tf.random_uniform([vocab_size, rnn_size], -1.0, 1.0))
    get_embedding('./checkpoint/news', sess)
    print embedding.eval(sess)
