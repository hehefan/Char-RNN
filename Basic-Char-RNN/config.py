import tensorflow as tf

tf.app.flags.DEFINE_integer("num_units", 512, "Number of units of RNN cell.")
tf.app.flags.DEFINE_boolean("use_lstm", True, "LSTM or GRU for the model.")
tf.app.flags.DEFINE_boolean("layer_norm", False, "If `True`, layer normalization will be applied.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Unit Tensor or float between 0 and 1 representing the recurrent dropout probability value. If float and 1.0, no dropout will be applied.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_float("learning_rate", 0.002, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("sequence_length", 50, "RNN sequence length.")
tf.app.flags.DEFINE_string("data_dir", "../data/ptb/", "Data directory.")
tf.app.flags.DEFINE_string("checkpoint_dir", "CheckPoint", "Checkpoint directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("num_epochs", 50, "Number of epochs.")

FLAGS = tf.app.flags.FLAGS

