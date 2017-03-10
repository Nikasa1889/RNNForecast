import tensorflow as tf
import numpy as np


class ESNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_inputs, num_units, output_size, batch_size, leaking_rate):
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._output_size = output_size
        self._leaking_rate = leaking_rate
        self.batch_size = batch_size
        print('num_inputs: %d. num_units: %d' % (num_inputs, num_units))
        np.random.seed(42)
        Win = (np.random.rand(1+num_inputs, num_units)-0.5) * 1
        W = np.random.rand(num_units, num_units)-0.5
        print('Computing spectral radius...')
        rhoW = max(abs(np.linalg.eig(W)[0]))
        print('done.')
        W *= 1.25 / rhoW
        self.Win = tf.constant(Win, name='Win', dtype=tf.float32)
        self.Wr = tf.constant(W, name='Wr', dtype=tf.float32)

    @property
    def num_units(self):
        return self._num_units

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def output_size(self):
        return self._output_size

    @property
    def leaking_rate(self):
        return self._leaking_rate

    @property
    def state_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope('ESNCell', initializer=tf.random_uniform_initializer(-0.5, 0.5)):
            # init Win, W
            # Win = tf.get_variable('Win', shape=[1 + self.num_inputs, self.num_units], trainable=False)
            # Wr = tf.get_variable('Wr', shape=[self.num_units, self.num_units], trainable=False)
            a = self.leaking_rate
            bias_col = tf.ones((self.batch_size, 1), name='bias_col')
            x = (1 - a) * state + a * tf.tanh(tf.matmul(tf.concat(1, [inputs, bias_col]), self.Win)
                                              + tf.matmul(state, self.Wr))
            y = tf.nn.rnn_cell._linear([bias_col, inputs, x], self.output_size, True)
        return y, x