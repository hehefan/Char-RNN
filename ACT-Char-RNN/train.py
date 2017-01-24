from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model
from config import *

FLAGS = tf.app.flags.FLAGS

def train():
  data_loader = TextLoader(FLAGS.data_dir, FLAGS.batch_size, FLAGS.sequence_length)
  vocab_size = data_loader.vocab_size

  model = Model(FLAGS.num_units, FLAGS.use_lstm, FLAGS.epsilon, FLAGS.max_computation, FLAGS.time_penalty, vocab_size, FLAGS.sequence_length, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.max_gradient_norm)

  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  with tf.Session() as sess:
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print("Created model with fresh parameters.")
      sess.run(tf.global_variables_initializer())
    current_step = 0
    previous_losses = []
    for e in range(FLAGS.num_epochs):
      sess.run(model.learning_rate_decay_op)
      data_loader.reset_batch_pointer()
      state = np.zeros((FLAGS.batch_size, model.cell.state_size))
      for b in range(data_loader.num_batches):
        start = time.time()
        x, y = data_loader.next_batch()
        feed = {model.inputs: x, model.targets: y, model.initial_state: state}
        loss1, loss2, state, _, remainders, iterations = sess.run([model.loss1, model.loss2, model.final_state, model.update, model.remainders, model.iterations], feed)
        train_loss = loss1 + loss2
        end = time.time()
        print("{}/{} (epoch {}), loss1 = {:.3f}, loss2 = {:.3f}, learning_rate = {:f}, time/batch = {:.3f}".format(e * data_loader.num_batches + b, FLAGS.num_epochs * data_loader.num_batches, e, loss1, loss2, model.learning_rate.eval(), end - start))
        current_step += 1
        if len(previous_losses) > 10 and train_loss > max(previous_losses[-10:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(train_loss)
        if current_step % FLAGS.steps_per_checkpoint == 0  or (e==FLAGS.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
          R = ['%.3f'%remainders[j][0] for j in xrange(FLAGS.sequence_length)]
          I = ['%d'%iterations[j][0] for j in xrange(FLAGS.sequence_length)]
          print(' '.join(R))
          print(' '.join(I))
          checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
  train()
