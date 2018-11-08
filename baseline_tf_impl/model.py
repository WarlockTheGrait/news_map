import tensorflow as tf

from tensorflow.python.layers import base as base_layer
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.eager import context


class MyCell(tf.contrib.rnn.LayerRNNCell):
    def __init__(self,
                 num_descs,
                 alpha=0.5,
                 reuse=None,
                 name=None,
                 dtype=None,
                 **kwargs):

        super(MyCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn("%s: Note that this cell is not optimized for performance on GPU.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_descs = num_descs
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

        self.built = True

    def call(self, inputs, state):

        rnn_input = tf.matmul(tf.concat([inputs, state], 1), self._rnn_w)

        d_new = tf.cond(tf.equal(tf.count_nonzero(state), 0),
                        true_fn=lambda: tf.nn.softmax(rnn_input),
                        false_fn=lambda: self._alpha * tf.nn.softmax(rnn_input) + (1 - self._alpha) * state)

        return d_new, d_new

    def get_config(self):

        config = {
            "num_descs": self._num_descs,
            "alpha": self._alpha,
            "reuse": self._reuse,
        }
        base_config = super(MyCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Model(object):
    def __init__(self, config, batch, real_emb=None, neg_embs=None, is_training=True):
        d_words = config.d_words
        num_descs = config.num_descs
        batch_size = tf.shape(batch)[0]
        batch_len = tf.shape(batch)[1]

        inputs = tf.reshape(batch, [-1, d_words])
        first_w = tf.get_variable("first_w", [d_words, d_words], dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer())
        first_b = tf.get_variable("first_b", [d_words], dtype=tf.float32, initializer=tf.initializers.zeros())
        h = tf.nn.xw_plus_b(inputs, first_w, first_b)
        h = tf.nn.relu(h)

        h = tf.reshape(h, [batch_size, batch_len, d_words])

        # create rnn cell to be unrolled
        cell = MyCell(num_descs, alpha=config.alpha)
        # add a dropout wrapper if training
        if is_training and config.dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1 - config.dropout)

        seq_batch_len = self.seq_lenght(batch)

        self.descs, _ = tf.nn.dynamic_rnn(cell, h,
                                          sequence_length=seq_batch_len,
                                          dtype=tf.float32)
        d = tf.reshape(self.descs, [-1, num_descs])

        R = tf.get_variable("R", [num_descs, d_words], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        self.norm_R = tf.nn.l2_normalize(R, axis=1)

        if not is_training:
            return

        real_value, descs = self.reduce_zeros(tf.reshape(real_emb, [-1, d_words]), d)

        r = tf.matmul(descs, R)
        r = tf.nn.l2_normalize(r, axis=-1)

        shape = tf.shape(real_value)[0]
        addend = tf.ones(shape) - self.dot_fn(real_value, r)

        def fn(x): return tf.math.maximum(tf.zeros(shape),
                                          self.dot_fn(x, r) + addend)
        hingle_over_span = tf.map_fn(fn, neg_embs)
        mean_hingle = tf.reduce_mean(hingle_over_span)
        hingle = tf.reduce_sum(hingle_over_span) / tf.cast(batch_size, dtype=tf.float32)

        penalty = config.eps * tf.reduce_sum(
            (tf.matmul(self.norm_R, self.norm_R, transpose_b=True) - tf.eye(num_descs)) ** 2)

        # Update the cost
        #self.cost = hingle + penalty
        self.cost = mean_hingle + penalty

        mean_squared_error = tf.losses.mean_squared_error(r, tf.nn.l2_normalize(real_value))

        tf.summary.scalar('mean_squared_err', tf.reduce_mean(mean_squared_error))
        tf.summary.scalar('mean_hingle', mean_hingle)
        tf.summary.scalar('hingle', hingle)
        tf.summary.scalar('penalty', penalty)
        tf.summary.scalar('cost', self.cost)
        self.merged = tf.summary.merge_all()

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)

        grads, _ = tf.clip_by_global_norm(grads, 5)
        optimizer = tf.train.AdamOptimizer(config.lr)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

    def seq_lenght(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def reduce_zeros(self, matrix, matrix2):
        intermediate_tensor = tf.reduce_sum(tf.abs(matrix), 1)
        zero_vector = tf.zeros(shape=(1, 1), dtype=tf.float32)
        bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
        bool_mask = tf.reshape(bool_mask, shape=[-1])
        omit_zeros = tf.boolean_mask(matrix, bool_mask)
        omit_zeros2 = tf.boolean_mask(matrix2, bool_mask)
        return omit_zeros, omit_zeros2

    def dot_fn(self, x, y):
        return tf.reduce_sum(tf.multiply(x, y), 1)
