from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model
from model_config import *

def weighted_pick(weights):
  t = np.cumsum(weights)
  s = np.sum(weights)
  return(int(np.searchsorted(t, np.random.rand(1)*s)))

def sample(step=None, num=200, prime='we ', sampling_type=1):
  data_loader = TextLoader(FLAGS.data_dir, FLAGS.batch_size, FLAGS.sequence_length)
  vocab_size = data_loader.vocab_size
  chars = data_loader.chars
  vocab = data_loader.vocab

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
      
      state = np.zeros((1, model.cell.state_size))
      for char in prime[:-1]:
        x = np.zeros((1, 1))
        x[0, 0] = vocab[char]
        feed = {model.inputs: x, model.initial_state:state}
        [state] = sess.run([model.final_state], feed)

      ret = prime
      char = prime[-1]
      for n in range(num):
        x = np.zeros((1, 1))
        x[0, 0] = vocab[char]
        feed = {model.inputs: x, model.initial_state:state}
        [prob, state] = sess.run([model.probs, model.final_state], feed)
        p = prob[0]
        if sampling_type == 0:
          sample = np.argmax(p)
        elif sampling_type == 2:
          if char == ' ':
            sample = weighted_pick(p)
          else:
            sample = np.argmax(p)
        else:
          sample = weighted_pick(p)
          
        sample = weighted_pick(p)
        pred = chars[sample]
        ret += pred
        char = pred
      print(ret)

if __name__ == '__main__':
  sample()
