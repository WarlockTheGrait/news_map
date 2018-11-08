import tensorflow as tf
import random
import datetime as dt

from model import Model
from generate_datasets import generate_train_datasets
from load_data import load_data, load_We

MODEL_NAME = 'mean_loss-20-batch_size-clip_grads'

data_path = 'models'


class Config:
    # number of descriptors
    num_descs = 20

    # number of negative samples per relationship (for loss)
    num_negs = 50

    # word dropout probability
    word_dropout = 0.75
    batch_size = 20
    alpha = 0.5
    dropout = 1

    epoch_size = 15
    lr = 0.001
    eps = 1e-6


def train(model_save_name, print_iter=50):
    config = Config()
    print('Loading data')
    We = load_We()    # Word embeddings matrix
    span_dict, span_info, _ = load_data(r'data/relationships.csv.gz', r'data/metadata.pkl')
    print('Data loaded')
    config.d_words = We.shape[1]
    config.num_traj = len(span_dict)
    config.num_traj = len(span_dict)

    keys = list(span_dict.keys())
    random.shuffle(keys)

    # setup data and models
    batch, real_emb, neg_embs = generate_train_datasets(span_dict, keys, We, config)
    m = Model(config, batch, real_emb=real_emb, neg_embs=neg_embs, is_training=True)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(data_path + '/logs', sess.graph)
        sess.run([init_op])

        saver = tf.train.Saver()
        glob_step = 0
        curr_time = dt.datetime.now()
        for epoch in range(config.epoch_size):
            print("Epoch {} started".format(epoch))
            for step in range(config.num_traj // config.batch_size + 1):
                cost, _, merged = sess.run([m.cost, m.train_op, m.merged])
                glob_step += 1
                train_writer.add_summary(merged, glob_step)

                if step % print_iter == print_iter-1:
                    seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = dt.datetime.now()
                    print("Epoch {}, Step {}, cost: {:.3f}, Seconds per step: {:.3f}".format(epoch,
                                                                                             step + 1, cost, seconds))

            # save a model checkpoint
            # saver.save(sess, data_path + '\\' + model_save_name, global_step=epoch)
        # do a final save
        saver.save(sess, data_path + '\\' + model_save_name + '-final')


if __name__ == '__main__':
    train(model_save_name=MODEL_NAME)

