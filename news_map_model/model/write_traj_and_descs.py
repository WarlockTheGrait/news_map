import csv
import numpy as np
import tensorflow as tf
import datetime as dt
import pickle
import os
import fastText

from gensim.models.keyedvectors import KeyedVectors

from cashed_datasets import generate_test_datasets
from model import Model
from train import Config
# from utils.notification_bot import get_notification

from utils.enum_classes import WordModel


def get_We(config):
    with open('data\\words_set.pkl', 'rb') as f:
        words_set = pickle.load(f)
    words_list = list(words_set)
    revmap = {i: word for i, word in enumerate(words_list)}
    del words_set
    if config.words_model is WordModel.FASTTEXT:
        model = fastText.load_model(r'C:\Users\Peksha\fastText\ft_native_300_ru_wiki_lenta_lemmatize.bin')
        gen = (model.get_word_vector(word) for word in words_list)
        We = np.array([word / np.linalg.norm(word) for word in gen])
        print(We.shape)
        del model
    else:
        if config.words_model is WordModel.WORD2VEC:
            model = KeyedVectors.load_word2vec_format(
                'C:\\Users\\Peksha\\Word2vec\\ruscorpora_upos_skipgram_300_10_2017.bin.gz',
                binary=True)
            model.vocab = {k[0: k.index('_')]: v for k, v in model.vocab.items()}
            We = np.array([model[word] for word in words_list])
        else:
            raise Exception('Word model not supported')
    print('We norm ', np.mean(np.linalg.norm(We, axis=1)))
    return We, revmap


def compute_descriptors(R, We, revmap, num_descs, descriptor_log):
    log = open(descriptor_log, 'w', encoding='utf-8')
    for ind in range(num_descs):
        desc = R[ind] / np.linalg.norm(R[ind])
        sims = We.dot(desc.T)
        ordered_words = np.argsort(sims)[::-1]
        desc_list = [revmap[w] for w in ordered_words[:15]]
        log.write(' '.join(desc_list) + '\n')
        print('descriptor %d:' % ind)
        print(desc_list)
    log.flush()
    log.close()


# @get_notification()
def write_data(model_name, model_path, config, print_iter=50):

    descriptor_log = f'descriptors/{model_name}-desc.log'
    trajectory_log = f'trajectories/{model_name}-traj.log'

    R_fix = np.load('R-fix.npy')
    with tf.variable_scope(model_name):
        batch, keys, num_traj = generate_test_datasets(config)
        m = Model(config, batch,  is_training=False)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, model_path + '\\' + model_name)
            curr_descs, R = sess.run([m.descs, m.norm_R])

            tlog = open(trajectory_log, 'w', encoding='utf-8')
            traj_writer = csv.writer(tlog, lineterminator='\n')
            traj_writer.writerow(['News', 'Person', 'Article ID'] +
                                 ['Topic ' + str(i) for i in range(config.num_descs)])

            for step in range(num_traj // config.batch_size + 1):
                n = 0
                for i in range(config.batch_size * step, min(num_traj, config.batch_size * (step + 1))):
                    key = keys[i]
                    news, person = key.split('--')
                    traj = curr_descs[n]
                    n += 1
                    used = np.sign(np.max(np.abs(traj), axis=1))
                    length = np.sum(used, axis=0)
                    for ind in range(int(length)):
                        span_desc = traj[ind]
                        # print(np.sum(span_desc))
                        traj_writer.writerow([news, person, ind] +
                                             list(span_desc))

                if num_traj <= config.batch_size * (step + 1):
                    break
                curr_descs = sess.run([m.descs])
                curr_descs = np.array(curr_descs)
                curr_descs = np.reshape(curr_descs, (curr_descs.shape[-3], -1, config.num_descs))

            tlog.flush()
            tlog.close()

            We, revmap = get_We(config)

            compute_descriptors(R, We, revmap, config.num_descs, descriptor_log)


if __name__ == '__main__':
    model_path = 'saved_models'
    model_save_name = ''
    with open(f'{model_path}/config/{model_save_name}.pkl', 'rb') as f:
        config = pickle.load(f)

    config.is_training = False

    write_data(model_save_name, model_path, config)

