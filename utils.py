# -*- coding: utf-8 -*-

import os
import re
import codecs
import collections
import cPickle
import numpy as np


PAD = "_PAD"
GO = "_GO"
EOS = "_EOS"
UNK = "_UNK"
SPACE = " "
NEW_LINE = "\n"
START_VOCAB = [PAD, GO, EOS, UNK, SPACE, NEW_LINE]


def normalize_unicodes(text):
  text = normalize_punctuation(text)
  text = "".join([Q2B(c) for c in list(text)])
  return text


def replace_all(repls, text):
  # return re.sub('|'.join(repls.keys()), lambda k: repls[k.group(0)], text)
  return re.sub(u'|'.join(re.escape(key) for key in repls.keys()),
                lambda k: repls[k.group(0)], text)


def normalize_punctuation(text):
  cpun = [['	'],
          [u'﹗'],
          [u'“', u'゛', u'〃', u'′'],
          [u'”'],
          [u'´', u'‘', u'’'],
          [u'；', u'﹔'],
          [u'《', u'〈', u'＜'],
          [u'》', u'〉', u'＞'],
          [u'﹑'],
          [u'【', u'『', u'〔'],
          [u'】', u'』', u'〕'],
          [u'（', u'「'],
          [u'）', u'」'],
          [u'﹖'],
          [u'︰', u'﹕'],
          [u'・', u'．', u'·', u'･', u'‧'],
          [u'●', u'○', u'▲', u'◎', u'◇', u'■', u'□', u'※', u'◆'],
          [u'〜', u'～', u'∼'],
          [u'—', u'ー', u'―', u'‐', u'−', u'─', u'﹣', u'–']]
  epun = [u' ', u'！', u'"', u'"', u'\'', u';', u'<', u'>', u'、', u'[', u']', u'(', u')', u'？', u'：', u'･', u'•', u'~', u'-']
  repls = {}

  for i in xrange(len(cpun)):
    for j in xrange(len(cpun[i])):
      repls[cpun[i][j]] = epun[i]

  return replace_all(repls, text)


def Q2B(uchar):
  """全角转半角"""
  inside_code = ord(uchar)
  if inside_code == 0x3000:
    inside_code = 0x0020
  else:
    inside_code -= 0xfee0
  #转完之后不是半角字符返回原来的字符
  if inside_code < 0x0020 or inside_code > 0x7e:
    return uchar
  return unichr(inside_code)


class TextLoader():
  def __init__(self, data_dir, batch_size, seq_length, encoding="utf-8"):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.encoding = encoding

    input_file = os.path.join(data_dir, "input.txt")
    vocab_file = os.path.join(data_dir, "vocab.pkl")
    tensor_file = os.path.join(data_dir, "data.npy")

    if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
      print "reading text file"
      self.preprocess(input_file, vocab_file, tensor_file)
    else:
      print "loading preprocessed files"
      self.load_preprocessed(vocab_file, tensor_file)
    self.create_batches()
    self.reset_batch_pointer()

  def preprocess(self, input_file, vocab_file, tensor_file):
    with codecs.open(input_file, "r", encoding=self.encoding) as f:
      data = f.read()
      data = normalize_unicodes(data)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    self.chars, counts = zip(*count_pairs)
    threshold = 10
    self.chars = [c for i, c in enumerate(self.chars) if counts[i] > threshold]
    self.vocab_size = len(self.chars)
    self.vocab = dict(zip(self.chars, range(len(self.chars))))
    with open(vocab_file, 'wb') as f:
      cPickle.dump(self.chars, f)
    unk_index = START_VOCAB.index(UNK)
    self.tensor = np.array([self.vocab.get(c, unk_index) for c in data])
    np.save(tensor_file, self.tensor)

  def load_preprocessed(self, vocab_file, tensor_file):
    with open(vocab_file, 'rb') as f:
      self.chars = cPickle.load(f)
    self.vocab_size = len(self.chars)
    self.vocab = dict(zip(self.chars, range(len(self.chars))))
    self.tensor = np.load(tensor_file)
    self.num_batches = int(self.tensor.size / (self.batch_size *
                           self.seq_length))

  def create_batches(self):
    self.num_batches = int(self.tensor.size / (self.batch_size *
                           self.seq_length))

    # When the data (tensor) is too small, let's give them a better error message
    if self.num_batches==0:
      assert False, "Not enough data. Make seq_length and batch_size small."

    self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
    xdata = self.tensor
    ydata = np.copy(self.tensor)
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
    self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

  def next_batch(self):
    x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
    self.pointer += 1
    return x, y

  def reset_batch_pointer(self):
    self.pointer = 0
