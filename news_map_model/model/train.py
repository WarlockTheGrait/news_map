import datetime as dt
import os
import pickle

import tensorflow as tf

from model import Model
from cashed_datasets import generate_train_datasets
from utils.enum_classes import Rnn, WordModel


class Config:
    save_path = 'saved_models'
    data_path = os.path.dirname(os.getcwd()) + '\\data'

    num_descs = 20  # number of descriptors
    num_negs = 50  # number of negative samples per relationship (for loss)
    word_dropout = 0.75
    batch_size = 29

    cashed_data = True

    alpha = 1
    beta = 0.5

    epoch_size = 150
    lr = 0.001
    lambda_coef = 1e-5  # lambda coef for loss

    new_loss = True
    train_alpha = False

    words_model = WordModel.FASTTEXT
    rnn = Rnn.LSTM

    dataset_random_seed = 9

    scaled_lr = True  # experiments with varied learning rate (from Attention ia all you need)
    lr_coef = 0.05
    warmup_steps = 750.


def train(config, print_iter=5):
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    if not os.path.exists(config.save_path + '/configs'):
        os.makedirs(config.save_path + '/configs')

    if not os.path.exists(config.save_path + '/info'):
        os.makedirs(config.save_path + '/info')

    if config.new_loss:
        loss = 'newloss'
    else:
        loss = 'oldloss'

    if config.train_alpha:
        model_name = f'{config.rnn.value}--{config.words_model.value}--{loss}--alpha-trained'
    else:
        model_name = f'{config.rnn.value}--{config.words_model.value}--{loss}'

    print(model_name)

    with tf.variable_scope(model_name):
        # setup data and models
        print('Generate datasets')
        batch, real_emb, neg_embs, num_traj = generate_train_datasets(config)
        m = Model(config, batch, real_emb=real_emb, neg_embs=neg_embs, is_training=True)
        init_op = tf.global_variables_initializer()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            if not os.path.exists('tensorboard_logs'):
                os.makedirs('tensorboard_logs')
            train_writer = tf.summary.FileWriter('tensorboard_logs', sess.graph)

            sess.run([init_op])

            saver = tf.train.Saver()
            glob_step = 0
            curr_time = dt.datetime.now()

            num_steps = num_traj // config.batch_size \
                if num_traj % config.batch_size == 0 else num_traj // config.batch_size + 1
            for epoch in range(config.epoch_size):
                print("Epoch {} started".format(epoch))
                error = 0.
                difference = 0.
                pure_error = 0.
                for step in range(num_steps):

                    cost, _, merged, diff, pure = sess.run([m.cost, m.train_op, m.merged,
                                                            m.difference, m.pure_error],
                                                           {m.step_num: step + epoch * num_steps + 1})

                    glob_step += 1
                    train_writer.add_summary(merged, glob_step)
                    error += cost
                    difference += diff
                    pure_error += pure
                    if step % print_iter == print_iter - 1:
                        seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                        curr_time = dt.datetime.now()
                        print("Epoch {}, Step {}, cost: {:.3f}, Seconds per step: {:.2f}".format(epoch,
                                                                                                 step + 1, cost,
                                                                                                 seconds))
                error /= num_steps
                difference /= num_steps
                pure_error /= num_steps

                print(f'Epoch {epoch}, Error {error:.4f} Difference {difference:.3f} Pure error {pure_error:.3f}')

                # save a model checkpoint
                # if epoch % 100 == 0 and epoch != config.epoch_size - 1 and epoch != 0:
                    # saver.save(sess, save_path + '\\' + model_save_name)

            # do a final save
            print('Final results')
            error = 0.
            difference = 0.
            pure_error = 0.
            for step in range(num_steps):
                cost, diff, pure = sess.run([m.cost, m.difference, m.pure_error])
                error += cost
                difference += diff
                pure_error += pure
            error /= num_steps
            difference /= num_steps
            pure_error /= num_steps

            saver.save(sess, config.save_path + '\\' + model_name)
            
            result = f'{model_name}, Error {error:.4f} Difference {difference:.3f} Pure error {pure_error:.3f}'
            print(result)

            with open(f'{config.save_path}/config/{model_name}.pkl', 'wb') as f:
                pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
                
            with open(f'{config.save_path}/info/{model_name}.log', 'w', encoding='utf-8') as log:
                log.write(result + '\n')

            return result


if __name__ == '__main__':
    config = Config()
    train(config)
