from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

tf.app.flags.DEFINE_integer("num_units", 128, "Number of units of RNN cell.")
tf.app.flags.DEFINE_boolean("use_lstm", True, "LSTM or GRU for the model.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_float("learning_rate", 0.002, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.97, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("sequence_length", 50, "RNN sequence length.")
tf.app.flags.DEFINE_string("data_dir", "data/ptb/", "Data directory.")
tf.app.flags.DEFINE_string("checkpoint_dir", "CheckPoint", "Checkpoint directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("num_epochs", 50, "Number of epochs.")

FLAGS = tf.app.flags.FLAGS

def weighted_pick(weights):
  t = np.cumsum(weights)
  s = np.sum(weights)
  return(int(np.searchsorted(t, np.random.rand(1)*s)))

def sample(step=None, num=200, prime='we ', sampling_type=1):
  data_loader = TextLoader(FLAGS.data_dir, FLAGS.batch_size, FLAGS.sequence_length)
  vocab_size = data_loader.vocab_size
  chars = data_loader.chars
  vocab = data_loader.vocab

  model = Model(FLAGS.num_units, FLAGS.use_lstm, FLAGS.num_layers, vocab_size, 1, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.max_gradient_norm, True)

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
