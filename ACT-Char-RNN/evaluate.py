from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle
import codecs

from utils import TextLoader
from model import Model
from config import *
import sys

def evaluate_one(step=None):
  input_file = os.path.join(FLAGS.data_dir, "ptb.valid.txt")
  with codecs.open(input_file, "r", encoding='utf-8') as f:
    data = f.read()

  data_loader = TextLoader(FLAGS.data_dir, FLAGS.batch_size, FLAGS.sequence_length)
  vocab_size = data_loader.vocab_size
  chars = data_loader.chars
  vocab = data_loader.vocab

  data = [vocab[char] for char in data]

  model = Model(FLAGS.num_units, FLAGS.use_lstm, FLAGS.layer_norm, 1.0, FLAGS.num_layers, vocab_size, 1, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.max_gradient_norm, True)
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if step == None:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        ckpt_path = os.path.join(FLAGS.checkpoint_dir,'ckpt-%d'%step)
        print("Reading model parameters from %s" % ckpt_path)
        model.saver.restore(sess, ckpt_path)
      
      total = 0
      state = np.zeros((1, model.cell.state_size))
      for i in xrange(len(data)-1):
        x = np.zeros((1, 1))
        x[0, 0] = data[i]
        feed = {model.inputs: x, model.initial_state:state}
        [prob, state] = sess.run([model.probs, model.final_state], feed)
        p = prob[0][data[i+1]]
        total -= np.log2(p)
  print(total/(len(data)-1))

def evaluate():
  input_file = os.path.join(FLAGS.data_dir, "ptb.test.txt")
  with codecs.open(input_file, "r", encoding='utf-8') as f:
    data = f.read()

  data_loader = TextLoader(FLAGS.data_dir, FLAGS.batch_size, FLAGS.sequence_length)
  vocab_size = data_loader.vocab_size
  chars = data_loader.chars
  vocab = data_loader.vocab

  data = [vocab[char] for char in data]
  model = Model(FLAGS.num_units, FLAGS.use_lstm, FLAGS.layer_norm, 1.0, FLAGS.num_layers, vocab_size, 1, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.max_gradient_norm, True)
  max_bpc = 0
  with tf.Session() as sess:
    step = 0
    while True:
      step += FLAGS.steps_per_checkpoint
      ckpt_path = os.path.join(FLAGS.checkpoint_dir,'ckpt-%d'%step)
      if os.path.isfile(ckpt_path + '.index'):
        model.saver.restore(sess, ckpt_path)
        total = 0
        state = np.zeros((1, model.cell.state_size))
        for i in xrange(len(data)-1):
          x = np.zeros((1, 1))
          x[0, 0] = data[i]
          feed = {model.inputs: x, model.initial_state:state}
          [prob, state] = sess.run([model.probs, model.final_state], feed)
          p = prob[0][data[i+1]]
          total -= np.log2(p)
        bpc = total/(len(data)-1)
        if bpc > max_bpc:
          max_bpc = bpc
        print('STEP %d: %.3f'%(step, bpc))
        sys.stdout.flush()
      else:
        break
if __name__ == '__main__':
  evaluate()
