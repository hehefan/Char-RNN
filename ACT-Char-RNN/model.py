import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import layers

import numpy as np

class LayerNormBasicLSTMCell(rnn_cell.RNNCell):
  def __init__(self, num_units, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=math_ops.tanh, layer_norm=True, norm_gain=1.0, norm_shift=0.0, dropout_keep_prob=1.0, dropout_prob_seed=None):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._g = norm_gain
    self._b = norm_shift

  @property
  def state_size(self):
    return (rnn_cell.LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._g)
    beta_init = init_ops.constant_initializer(self._b)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args):
    out_size = 4 * self._num_units
    proj_size = args.get_shape()[-1]
    weights = vs.get_variable("weights", [proj_size, out_size])
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("biases", [out_size])
      out = nn_ops.bias_add(out, bias)
    return out

  def __call__(self, inputs, state, scope=None):
    """LSTM cell with layer normalization and recurrent dropout."""

    with vs.variable_scope(scope or "layer_norm_basic_lstm_cell"):
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = tf.split(split_dim=1, num_split=2, value=state)

      args = array_ops.concat_v2([inputs, h], 1)
      concat = self._linear(args)

      i, j, f, o = tf.split(split_dim=1, num_split=4, value=concat)
      if self._layer_norm:
        i = self._norm(i, "input")
        j = self._norm(j, "transform")
        f = self._norm(f, "forget")
        o = self._norm(o, "output")

      g = self._activation(j)
      if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
        g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

      new_c = (c * math_ops.sigmoid(f + self._forget_bias)
               + math_ops.sigmoid(i) * g)
      if self._layer_norm:
        new_c = self._norm(new_c, "state")
      new_h = self._activation(new_c) * math_ops.sigmoid(o)
      
      if self._state_is_tuple:
        new_state = rnn_cell.LSTMStateTuple(new_c, new_h)
      else:
        new_state = array_ops.concat_v2([new_c, new_h], 1)
      return new_h, new_state

class Model(object):
  def __init__(self, num_units, use_lstm, layer_norm, dropout_keep_prob, num_layers, vocab_size, sequence_length, learning_rate, learning_rate_decay_factor, max_gradient_norm, forward_only=False):
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False) 

    single_cell = tf.nn.rnn_cell.GRUCell(num_units)
    if use_lstm:
      if layer_norm:
        single_cell = LayerNormBasicLSTMCell(num_units=num_units, dropout_keep_prob=dropout_keep_prob, state_is_tuple=False)
      else:
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=False)
    cell = single_cell
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers, state_is_tuple=False)
    self.cell = cell

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

    outputs, self.final_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if forward_only else None, scope='rnnlm')
    outputs = tf.reshape(tf.concat(1, outputs), [-1, num_units])
    self.logits = tf.matmul(outputs, softmax_w) + softmax_b
    targets = tf.reshape(self.targets, [-1])
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=targets))
    self.loss = loss/tf.cast(tf.shape(targets)[0], loss.dtype)
  
    if forward_only:
      self.probs = tf.nn.softmax(self.logits)
    else:
      params = tf.trainable_variables()
      clipped_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, params), max_gradient_norm)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=999999999)
