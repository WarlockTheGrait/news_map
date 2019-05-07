import tensorflow as tf

from rnn_layers import rnn_cell, LossCell


class Model(object):

    def __init__(self, config, batch, real_emb=None, neg_embs=None, is_training=True):
        d_words = config.d_words
        num_descs = config.num_descs
        batch_size = tf.shape(batch)[0]
        batch_len = tf.shape(batch)[1]

        inputs = tf.reshape(batch, [-1, d_words])
        first_w = tf.get_variable("first_w", [d_words, d_words], dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer())
        first_b = tf.get_variable("first_b", [d_words], dtype=tf.float32, initializer=tf.initializers.constant(0.1))
        h = tf.nn.xw_plus_b(inputs, first_w, first_b)
        h = tf.nn.relu(h)

        h = tf.reshape(h, [batch_size, batch_len, d_words])

        # create rnn cell to be unrolled
        cell = rnn_cell(config, num_descs)

        seq_batch_len = self.seq_lenght(batch)

        self.descs, _ = tf.nn.dynamic_rnn(cell, h,
                                          sequence_length=seq_batch_len,
                                          initial_state=cell.zero_state(batch_size, dtype=tf.float32),
                                          dtype=tf.float32)

        d = tf.reshape(self.descs, [-1, num_descs])

        R = tf.get_variable("R", [num_descs, d_words], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        self.norm_R = tf.nn.l2_normalize(R, axis=1)

        if not is_training:
            return

        if config.new_loss:
            loss_cell = LossCell(d_words, beta=config.beta)
            real_emb = tf.reshape(real_emb, [batch_size, batch_len, d_words])
            real_emb, _ = tf.nn.dynamic_rnn(loss_cell, real_emb,
                                            sequence_length=seq_batch_len,
                                            dtype=tf.float32)

        real_value, descs = self.reduce_zeros(tf.reshape(real_emb, [-1, d_words]), d)

        r = tf.matmul(descs, R)
        r = tf.nn.l2_normalize(r, axis=-1)

        shape = tf.shape(real_value)[0]
        addend = tf.ones(shape) - self.dot_fn(real_value, r)

        def loss_fn(x):
            return tf.math.maximum(tf.zeros(shape), self.dot_fn(x, r) + addend)

        hingle_over_span = tf.map_fn(loss_fn, neg_embs)
        mean_hingle = tf.reduce_mean(hingle_over_span)

        penalty = config.eps * tf.reduce_sum(
            (tf.matmul(self.norm_R, self.norm_R, transpose_b=True) - tf.eye(num_descs)) ** 2)

        self.pure_error = tf.reduce_mean(addend)
        self.difference = mean_hingle - tf.reduce_mean(
            tf.map_fn(lambda x: tf.math.maximum(tf.zeros(shape), self.dot_fn(x, real_value)), neg_embs))

        # Update the cost
        self.cost = mean_hingle + penalty

        # tf.summary.scalar('mean_hingle', mean_hingle)
        tf.summary.scalar('penalty', penalty)
        tf.summary.scalar('pure_error', self.pure_error)
        tf.summary.scalar('difference', self.difference)
        tf.summary.scalar('cost', self.cost)
        self.merged = tf.summary.merge_all()

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)

        grads, _ = tf.clip_by_global_norm(grads, 5)
        if config.scaled_lr:
            self.step_num = tf.placeholder(tf.float32)
            lr = config.lr_coef * tf.minimum(tf.math.pow(self.step_num, tf.constant(-0.5)),
                                             self.step_num * (config.warmup_steps ** -1.5))
        else:
            lr = config.lr
        optimizer = tf.train.AdamOptimizer(lr)
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
