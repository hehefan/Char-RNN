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
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 5000, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("num_epochs", 50, "Number of epochs.")

FLAGS = tf.app.flags.FLAGS


def evaluate(step=None):
  input_file = os.path.join(FLAGS.data_dir, "ptb.valid.txt")
  with codecs.open(input_file, "r", encoding='utf-8') as f:
    data = f.read()

  data_loader = TextLoader(FLAGS.data_dir, FLAGS.batch_size, FLAGS.sequence_length)
  vocab_size = data_loader.vocab_size
  chars = data_loader.chars
  vocab = data_loader.vocab

  data = [vocab[char] for char in data]

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

if __name__ == '__main__':
  evaluate()
