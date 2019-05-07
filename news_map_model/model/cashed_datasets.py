import tensorflow as tf
import numpy as np
import os
import pickle
import random

from utils.enum_classes import WordModel


def decode(filename, config):
    features = tf.parse_single_example(
        filename,
        features={
            'raw': tf.FixedLenFeature([], tf.string),
        })
    emb = tf.decode_raw(features['raw'], tf.float32)
    emb = tf.reshape(emb, [-1, config.d_words])
    return emb


def neg_decode(filename):
    features = tf.parse_single_example(
        filename,
        features={
            'raw': tf.FixedLenFeature([], tf.string),
        })
    emb = tf.decode_raw(features['raw'], tf.float32)
    return emb


def train_datasets(batch_filenames, real_filenames, neg_filenames, config):
    batch_dataset = tf.data.TFRecordDataset(batch_filenames) \
        .map(lambda x: decode(x, config), num_parallel_calls=4) \
        .repeat() \
        .padded_batch(config.batch_size, padded_shapes=[None, None]) \
        .prefetch(1)

    real_dataset = tf.data.TFRecordDataset(real_filenames) \
        .map(lambda x: decode(x, config), num_parallel_calls=4) \
        .repeat() \
        .padded_batch(config.batch_size, padded_shapes=[None, None]) \
        .prefetch(1)

    neg_dataset = tf.data.TFRecordDataset(neg_filenames) \
        .map(neg_decode, num_parallel_calls=4) \
        .repeat() \
        .batch(config.num_negs) \
        .prefetch(1)

    batch = batch_dataset.make_one_shot_iterator().get_next(name='batch')
    real_emb = real_dataset.make_one_shot_iterator().get_next(name='real')
    neg_emb = neg_dataset.make_one_shot_iterator().get_next(name='neg')

    return batch, real_emb, neg_emb


def test_dataset(real_filenames, config):
    real_dataset = tf.data.TFRecordDataset(real_filenames) \
        .map(lambda x: decode(x, config), num_parallel_calls=4) \
        .repeat() \
        .padded_batch(config.batch_size, padded_shapes=[None, None]) \
        .prefetch(1)

    real_emb = real_dataset.make_one_shot_iterator().get_next(name='real')

    return real_emb


def generate_train_datasets(config):
    data_folder = config.data_path
    if config.words_model is WordModel.WORD2VEC:
        config.d_words = 300
    elif config.words_model is WordModel.TWITTER_FASTTEXT:
        config.d_words = 100
    elif config.words_model is WordModel.FASTTEXT:
        config.d_words = 300
    elif config.words_model is WordModel.ELMO:
        config.d_words = 1024
    else:
        raise Exception('Unknown words model')

    news_names = ['Lenta', 'Meduza', 'Rain', 'Ria']
    tuple_filenames = []

    filepath = f'{data_folder}\\{config.words_model.value}'
    nfile = f'{data_folder}\\{config.words_model.value}\\neg'

    random.seed(config.dataset_random_seed)

    for news in news_names:
        _, _, files = next(os.walk(f'{filepath}\\{news}\\batch{config.word_dropout}'), (None, None, []))

        batch_filenames = [f'{filepath}\\{news}\\batch{config.word_dropout}\\{file}' for file in files]
        real_filenames = [f'{filepath}\\{news}\\real\\{file}' for file in files]

        tuple_filenames.extend([(real_filenames[i], batch_filenames[i]) for i in range(len(files))])

    num_traj = len(tuple_filenames)

    random.shuffle(tuple_filenames)
    real_filenames = [tuple_filenames[i][0] for i in range(num_traj)]
    batch_filenames = [tuple_filenames[i][1] for i in range(num_traj)]

    print('Number of traj ', num_traj)

    del tuple_filenames

    _, _, neg_files = next(os.walk(nfile), (None, None, []))
    neg_filenames = [f'{nfile}\\{file}' for file in neg_files]
    random.shuffle(neg_filenames)

    del neg_files

    batch, real_emb, neg_emb = train_datasets(batch_filenames, real_filenames, neg_filenames, config)

    return batch, real_emb, neg_emb, num_traj


def generate_test_datasets(config):
    data_folder = config.data_path
    if config.words_model is WordModel.WORD2VEC:
        config.d_words = 300
    elif config.words_model is WordModel.TWITTER_FASTTEXT:
        config.d_words = 100
    elif config.words_model is WordModel.FASTTEXT:
        config.d_words = 300
    elif config.words_model is WordModel.ELMO:
        config.d_words = 1024
    else:
        raise Exception('Unknown words model')

    news_names = ['Lenta', 'Meduza', 'Rain', 'Ria']

    filepath = f'{data_folder}/{config.words_model.value}'

    real_filenames = []
    keys = []
    for news in news_names:
        _, _, files = next(os.walk(f'{filepath}/{news}/real'), (None, None, []))

        real_filenames = [f'{filepath}/{news}/real/{file}' for file in files]
        keys.extend([f"{news}--{file.split('.')[0]}" for file in files])

    num_traj = len(real_filenames)

    print('Number of traj ', num_traj)

    real_emb = test_dataset(real_filenames, config)

    return real_emb, keys, num_traj
