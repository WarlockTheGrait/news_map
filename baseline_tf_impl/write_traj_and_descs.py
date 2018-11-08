import datetime as dt
import csv
import numpy as np
import tensorflow as tf

from model import Model
from generate_datasets import generate_test_dataset
from load_data import load_data, load_We
from train import Config, MODEL_NAME

data_path = 'models'
descriptor_log = 'descriptors/descriptors.log'
trajectory_log = 'trajectories/trajectories.log'


def compute_descriptors(R, We, revmap, num_descs):
    log = open(descriptor_log, 'w')
    for ind in range(num_descs):
        desc = R[ind] / np.linalg.norm(R[ind])
        sims = We.dot(desc.T)
        ordered_words = np.argsort(sims)[::-1]
        desc_list = [revmap[w] for w in ordered_words[:10]]
        log.write(' '.join(desc_list) + '\n')
        print('descriptor %d:' % ind)
        print(desc_list)
    log.flush()
    log.close()


def write_data(model_save_name=MODEL_NAME, print_iter=50):
    config = Config()
    print('Loading data')
    We = load_We()
    span_dict, span_info, revmap = load_data(r'data/relationships.csv.gz', r'data/metadata.pkl')
    print('Data loaded')
    config.d_words = We.shape[1]
    config.num_traj = len(span_dict)
    config.num_traj = len(span_dict)
    keys = list(span_dict.keys())

    # setup data and models
    batch = generate_test_dataset(span_dict, keys, We, config)
    m = Model(config, batch, is_training=False)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, data_path + '\\' + model_save_name + '-final')
        curr_descs, R = sess.run([m.descs, m.norm_R])

        compute_descriptors(R, We, revmap, config.num_descs)
        curr_time = dt.datetime.now()

        tlog = open(trajectory_log, 'w')
        traj_writer = csv.writer(tlog, lineterminator='\n')
        traj_writer.writerow(['Book', 'Char 1', 'Char 2', 'Span ID'] +
                             ['Topic ' + str(i) for i in range(config.num_descs)])

        for step in range(config.num_traj // config.batch_size + 1):
            n = 0
            for i in range(config.batch_size * step, min(config.num_traj, config.batch_size * (step + 1))):
                key = keys[i]
                book, c1, c2 = span_info[key]
                traj = curr_descs[n]
                n += 1
                used = np.sign(np.max(np.abs(traj), axis=1))
                length = np.sum(used, axis=0)
                for ind in range(int(length)):
                    span_desc = traj[ind]
                    traj_writer.writerow([book, c1, c2, ind] +
                                         list(span_desc))
            if step % print_iter == print_iter - 1:
                seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                curr_time = dt.datetime.now()
                print("Step {}, Seconds per step: {:.3f}".format(step + 1, seconds))

            if config.num_traj <= config.batch_size * (step + 1):
                break
            curr_descs = sess.run([m.descs])
            curr_descs = np.array(curr_descs)
            curr_descs = np.reshape(curr_descs, (curr_descs.shape[-3], -1, config.num_descs))

        tlog.flush()
        tlog.close()


if __name__ == '__main__':
    write_data()
