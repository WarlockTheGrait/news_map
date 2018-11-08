import tensorflow as tf
import numpy as np


def to_embedding(span, d_words, We, is_dropout=False, drop_prob=0):
    span_size = span.shape[0]
    i = 0
    emb = np.zeros(d_words)
    while i < span_size and span[i] != -1:
        if not is_dropout or np.random.random_sample() < 1 - drop_prob:
            emb += We[span[i]]

        i += 1
    return emb / max(np.linalg.norm(emb), 10e-6)


def generate_negative_samples(span_dict, keys, We, d_words, num_traj, num_negs=50):
    def f(i): return to_embedding(span_dict[keys[i]][np.random.randint(low=0, high=span_dict[keys[i]].shape[0])],
                                  d_words, We)

    inds = np.random.randint(0, num_traj, size=num_negs)
    lst_embs = np.array([f(i) for i in inds])
    return lst_embs


def train_generator(span_dict, keys, We, config):
    for key in keys:
        train_emb = np.array([to_embedding(span,
                                           d_words=config.d_words,
                                           We=We,
                                           is_dropout=True,
                                           drop_prob=config.word_dropout)
                              for span in span_dict[key]])
        yield train_emb


def real_generator(span_dict, keys, We, config):
    for key in keys:
        real_emb = np.array([to_embedding(span, d_words=config.d_words, We=We) for span in span_dict[key]])
        yield real_emb


def neg_generator(span_dict, keys, We, config):
    while True:
        neg_embs = generate_negative_samples(span_dict,
                                             keys,
                                             We,
                                             d_words=config.d_words,
                                             num_traj=config.num_traj,
                                             num_negs=config.num_negs)
        yield neg_embs


def generate_train_datasets(span_dict, keys, We, config):

    dataset_train = tf.data.Dataset.from_generator(lambda: train_generator(span_dict, keys, We, config),
                                                   tf.float32,
                                                   tf.TensorShape([None, config.d_words]))
    dataset_train = dataset_train.padded_batch(config.batch_size, padded_shapes=[None, None])
    dataset_train = dataset_train.repeat(count=config.epoch_size)
    dataset_train = dataset_train.prefetch(buffer_size=1)
    batch = dataset_train.make_one_shot_iterator().get_next(name='batch')

    dataset_real = tf.data.Dataset.from_generator(lambda: real_generator(span_dict, keys, We, config),
                                                  tf.float32,
                                                  tf.TensorShape([None, config.d_words]))
    dataset_real = dataset_real.padded_batch(config.batch_size, padded_shapes=[None, None])
    dataset_real = dataset_real.repeat(count=config.epoch_size)
    dataset_real = dataset_real.prefetch(buffer_size=1)
    real_emb = dataset_real.make_one_shot_iterator().get_next(name='real')

    dataset_neg = tf.data.Dataset.from_generator(lambda: neg_generator(span_dict, keys, We, config),
                                                 tf.float32,
                                                 tf.TensorShape([config.num_negs, config.d_words]))
    dataset_neg = dataset_neg.prefetch(buffer_size=1)
    neg_embs = dataset_neg.make_one_shot_iterator().get_next(name='neg')

    return batch, real_emb, neg_embs


def generate_test_dataset(span_dict, keys, We, config):
    dataset_real = tf.data.Dataset.from_generator(lambda: real_generator(span_dict, keys, We, config),
                                                  tf.float32,
                                                  tf.TensorShape([None, config.d_words]))
    dataset_real = dataset_real.padded_batch(config.batch_size, padded_shapes=[None, None])
    dataset_real = dataset_real.prefetch(buffer_size=1)
    real_emb = dataset_real.make_one_shot_iterator().get_next(name='real')

    return real_emb


