import tensorflow as tf

from tensorflow.python.layers import base as base_layer
from tensorflow.python.keras.utils import tf_utils

from utils.enum_classes import Rnn


class LossCell(tf.contrib.rnn.LayerRNNCell):
    def __init__(self,
                 d_words,
                 beta=0.5,
                 reuse=None,
                 name=None,
                 dtype=None,
                 **kwargs):

        super(LossCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._d_words = d_words
        self._beta = beta
        self._reuse = reuse

    @property
    def state_size(self):
        return self._d_words

    @property
    def output_size(self):
        return self._d_words

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        self.built = True

    def call(self, inputs, state):
        real_value = tf.cond(tf.equal(tf.count_nonzero(state), 0),
                             true_fn=lambda: inputs,
                             false_fn=lambda: (1 - self._beta) * inputs + self._beta * state)
        return real_value, real_value


class AlphaCell(tf.contrib.rnn.LayerRNNCell):
    def __init__(self, num_descs, alpha=1, train_alpha=False, reuse=None, name=None, dtype=None, **kwargs):

        super(AlphaCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self.alpha = alpha
        self._train_alpha = train_alpha
        self.num_descs = num_descs

    @property
    def state_size(self):
        return self.num_descs

    @property
    def output_size(self):
        return self.num_descs

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        self._w = self.add_variable(
            'alpha_weigths',
            shape=[self.num_descs, self.num_descs],
            initializer=tf.initializers.identity())

        if self._train_alpha:
            self._alpha = tf.get_variable('alpha', shape=[], dtype=tf.float32,
                                          initializer=tf.initializers.constant(0))
            self.alpha = tf.math.sigmoid(self._alpha)

        self.built = True

    def call(self, h, state):
        super_h = tf.matmul(h, self._w)
        coef = tf.reduce_sum(state, axis=1)
        another_coef = tf.expand_dims(tf.ones(tf.shape(coef)) - coef, 1)
        soft = tf.nn.softmax(super_h)

        if self._train_alpha:
            h_new = self.alpha * soft + (tf.ones([]) - self.alpha) * (state + tf.multiply(another_coef, soft))
        else:
            h_new = self.alpha * soft + (1 - self.alpha) * (state + tf.multiply(another_coef, soft))
        return h_new, h_new


class SimpleCell(tf.contrib.rnn.LayerRNNCell):
    def __init__(self,
                 num_descs,
                 alpha=0.5,
                 train_alpha=False,
                 reuse=None,
                 name=None,
                 dtype=None,
                 **kwargs):

        super(SimpleCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_descs = num_descs
        self._train_alpha = train_alpha
        self._alpha = alpha
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_descs

    @property
    def output_size(self):
        return self._num_descs

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[-1]
        self._rnn_w = self.add_variable(
            'rnn_weigths',
            shape=[input_depth + self._num_descs, self._num_descs],
            initializer=tf.glorot_uniform_initializer())

        if self._train_alpha:
            self._alpha = tf.get_variable('alpha', shape=[], dtype=tf.float32,
                                          initializer=tf.initializers.constant(self._alpha))
            self.alpha = tf.math.sigmoid(self._alpha)
        self.built = True

    def call(self, inputs, state):
        rnn_input = tf.matmul(tf.concat([inputs, state], 1), self._rnn_w)
        coef = tf.reduce_sum(state, axis=1)
        another_coef = tf.expand_dims(tf.ones(tf.shape(coef)) - coef, 1)
        soft = tf.nn.softmax(rnn_input)
        if self._train_alpha:
            d_new = self.alpha * soft + (tf.ones([]) - self.alpha) * (state + tf.multiply(another_coef, soft))
        else:
            d_new = self._alpha * soft + (1 - self._alpha) * (state + tf.multiply(another_coef, soft))
        return d_new, d_new

    def get_config(self):

        config = {
            "num_descs": self._num_descs,
            "alpha": self._alpha,
            "reuse": self._reuse,
        }
        base_config = super(SimpleCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def rnn_cell(config, num_descs):
    if config.rnn is Rnn.SIMPLE:
        cell = SimpleCell(num_descs, alpha=config.alpha, train_alpha=config.train_alpha)
    elif config.rnn is Rnn.GRU:
        gru = tf.contrib.rnn.GRUBlockCell(num_units=num_descs)

        alpha_cell = AlphaCell(num_descs=num_descs, alpha=config.alpha, train_alpha=config.train_alpha)

        cell = tf.nn.rnn_cell.MultiRNNCell([gru, alpha_cell])
    elif config.rnn is Rnn.LSTM:
        lstm = tf.contrib.rnn.LSTMBlockCell(num_units=num_descs)

        alpha_cell = AlphaCell(num_descs=num_descs, alpha=config.alpha, train_alpha=config.train_alpha)

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm, alpha_cell])
    else:
        print('Unknown rnn cell, simple cell is used')
        cell = SimpleCell(num_descs, alpha=config.alpha, train_alpha=config.train_alpha)

    return cell

