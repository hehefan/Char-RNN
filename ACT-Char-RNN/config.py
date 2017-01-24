import tensorflow as tf

tf.app.flags.DEFINE_integer("num_units", 1024, "Number of units of RNN cell.")
tf.app.flags.DEFINE_boolean("use_lstm", False, "LSTM or GRU for the model.")
tf.app.flags.DEFINE_float("epsilon", 0.01, "xx")
tf.app.flags.DEFINE_integer("max_computation", 10, "xxx")
tf.app.flags.DEFINE_float("time_penalty", 0.00001, "xx")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("sequence_length", 50, "RNN sequence length.")
tf.app.flags.DEFINE_string("data_dir", "../data/ptb/", "Data directory.")
tf.app.flags.DEFINE_string("checkpoint_dir", "CheckPoint", "Checkpoint directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs.")

FLAGS = tf.app.flags.FLAGS

