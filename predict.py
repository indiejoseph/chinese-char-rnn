import numpy as np
import tensorflow as tf

def sample(sess, model, chars, vocab, num=200, prime='The '):
  model.initial_state = model.cell.zero_state(1, tf.float32)

  # assign final state to rnn
  state_list = []
  for c, h in model.initial_state:
    state_list.extend([c.eval(), h.eval()])

  prime = prime.decode('utf-8')

  for char in prime[:-1]:
    x = np.zeros((1, 1))
    x[0, 0] = vocab.get(char, 0)
    feed = {model.input_data: x}
    fetchs = []
    for i in range(len(model.initial_state)):
      c, h = model.initial_state[i]
      feed[c], feed[h] = state_list[i*2:(i+1)*2]
    for c, h in model.final_state:
      fetchs.extend([c, h])
    state_list = sess.run(fetchs, feed)

  def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))

  ret = prime
  char = prime[-1]

  for _ in xrange(num):
    x = np.zeros((1, 1))
    x[0, 0] = vocab.get(char, 0)
    feed = {model.input_data: x}
    fetchs = [model.probs]
    for i in range(len(model.initial_state)):
      c, h = model.initial_state[i]
      feed[c], feed[h] = state_list[i*2:(i+1)*2]
    for c, h in model.final_state:
      fetchs.extend([c, h])
    res = sess.run(fetchs, feed)
    probs = res[0]
    state_list = res[1:]
    p = probs[0]
    # sample = int(np.random.choice(len(p), p=p))
    sample = weighted_pick(p)
    pred = chars[sample]
    ret += pred
    char = pred

  return ret
