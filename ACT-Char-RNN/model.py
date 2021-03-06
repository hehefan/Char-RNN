import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import layers
from act_cell import ACTCell 
import numpy as np


class Model(object):
  def __init__(self, num_units, use_lstm, epsilon, max_computation, time_penalty, vocab_size, sequence_length, learning_rate, learning_rate_decay_factor, max_gradient_norm, forward_only=False):
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False) 

    cell = tf.nn.rnn_cell.GRUCell(num_units)
    if use_lstm:
      cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=False)
    self.cell = ACTCell(cell, epsilon, max_computation)

    self.inputs = tf.placeholder(tf.int32, [None, sequence_length])
    self.targets = tf.placeholder(tf.int32, [None, sequence_length])
    self.initial_state = tf.placeholder(tf.float32, [None, cell.state_size])

    with tf.variable_scope('rnnlm'):
      softmax_w = tf.get_variable("softmax_w", [num_units, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])
      with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, num_units])
        inputs = tf.split(1, sequence_length, tf.nn.embedding_lookup(embedding, self.inputs))
        inputs = [tf.squeeze(i, [1]) for i in inputs]

    def loop(prev, _):
      prev = tf.matmul(prev, softmax_w) + softmax_b
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      return tf.nn.embedding_lookup(embedding, prev_symbol)

    outputs, self.final_state = seq2seq.rnn_decoder(inputs, self.initial_state, self.cell, loop_function=loop if forward_only else None, scope='rnnlm')
    self.remainders = self.cell.ACT_remainder
    self.iterations = self.cell.ACT_iterations

    outputs = tf.reshape(tf.concat(1, outputs), [-1, num_units])
    self.logits = tf.matmul(outputs, softmax_w) + softmax_b
    targets = tf.reshape(self.targets, [-1])
    self.loss1 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=targets))/tf.cast(tf.shape(targets)[0], tf.float32)
    self.loss2 = self.cell.calculate_ponder_cost()
    self.loss = self.loss1 + time_penalty*self.loss2
  
    if forward_only:
      self.probs = tf.nn.softmax(self.logits)
    else:
      params = tf.trainable_variables()
      clipped_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, params), max_gradient_norm)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=999999999)
