import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from cycler import cycler

from train import Config


# parse learned descriptors into a dict
def read_descriptors(desc_file):
    desc_map = {}
    f = open(desc_file, 'r', encoding='utf-8')
    for i, line in enumerate(f):
        line = line.split()
        desc_map[i] = line[0]
    return desc_map


# read learned trajectories file
def read_csv(csv_file):
    reader = csv.reader(open(csv_file, 'rt', encoding='utf-8'))
    all_traj = {}
    prev_news = None
    prev_person = None
    total_traj = 0
    for index, row in enumerate(reader):
        if index == 0:
            continue

        news, person = row[:2]
        if prev_news != news or prev_person != person:
            prev_news = news
            prev_person = person
            if news not in all_traj:
                all_traj[news] = {}
            all_traj[news][person] = []
            total_traj += 1

        else:
            all_traj[news][person].append(np.array(row[3:], dtype='float32'))

    print(len(all_traj), total_traj)
    return all_traj


# compute locations to write labels
# only write labels when the
def compute_centers(max_traj, smallest_shift):
    center_inds = []
    prev_topic = max_traj[0]
    tstart = 0
    for index, topic in enumerate(max_traj):
        if topic != prev_topic:
            center = int((index - tstart) / 2)
            if center > smallest_shift / 2:
                center_inds.append(tstart + center)
            tstart = index
            prev_topic = topic
    center = int((index - tstart) / 2)
    if index - tstart > smallest_shift:
        center_inds.append(tstart + center)

    return center_inds


def save_fig(news, rel, rtraj, fig_dir, smallest_shift):
    plt.close()
    rtraj_mat = np.array(rtraj)

    name = ' '.join([word.capitalize() for word in rel.split('___')[0].split('__')])

    plt.title(news + ': ' + name)
    plt.axis('off')

    max_rtraj = np.argmax(rtraj_mat, axis=1)
    rcenter_inds = compute_centers(max_rtraj, smallest_shift)

    for ind in range(0, len(max_rtraj)):
        topic = max_rtraj[ind]
        plt.axhspan(ind, ind + 1, 0.2, 0.4, color=color_list[topic])

        if ind in rcenter_inds:
            loc = (0.43, ind + 0.5)
            plt.annotate(rmn_descs[topic], loc, size=15,
                         verticalalignment='center',
                         color=color_list[topic])

    plt.xlim(0, 1.0)
    plt.arrow(0.1, 0, 0.0, len(rtraj),
              head_width=0.1, head_length=len(rtraj) / 12, lw=3,
              length_includes_head=True, fc='k', ec='k')

    props = {'ha': 'left', 'va': 'bottom', }
    plt.text(0.0, len(rtraj) / 2, 'TIME', props, rotation=90, size=15)
    props = {'ha': 'left', 'va': 'top', }

    if fig_dir is None:
        plt.show()
    else:
        fig_name = f'{fig_dir}/{news}/{rel}.png'
        print('figname = ', fig_name)
        plt.savefig(fig_name)


def viz_csv(rmn_traj, rmn_descs, color_list,
            min_length=10,
            smallest_shift=1, max_viz=True,
            fig_dir=None):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for news in rmn_traj:
        if not os.path.exists(f'{fig_dir}/{news}'):
            os.makedirs(f'{fig_dir}/{news}')
        for rel in rmn_traj[news]:
            rtraj = rmn_traj[news][rel]
            print(news, rel)
            if len(rtraj) <= 50:
                save_fig(news, rel, rtraj, fig_dir, smallest_shift)
            else:
                for i in range(len(rtraj) // 50):
                    save_fig(news, f'{rel} {i}', rtraj[i*50:50*(i+1)], fig_dir, smallest_shift)


if __name__ == '__main__':
    model_name = ''

    save_path = 'figs/final'

    rmn_descs = read_descriptors(f'descriptors/{model_name}-desc.log')
    rmn_traj = read_csv(f'trajectories/{model_name}-traj.log')

    print(rmn_descs)

    plt.style.use('ggplot')
    color_list = ["peru", "dodgerblue", "brown", "hotpink",
                  "aquamarine", "springgreen", "chartreuse", "fuchsia",
                  "mediumspringgreen", "burlywood", "midnightblue", "orangered",
                  "olive", "darkolivegreen", "darkmagenta", "mediumvioletred",
                  "darkslateblue", "saddlebrown", "darkturquoise", "cyan",
                  "chocolate", "cornflowerblue", "blue", "red",
                  "navy", "steelblue", "cadetblue", "forestgreen",
                  "black", "darkcyan"]
    color_list += color_list
    plt.rc('axes', prop_cycle=(cycler('color', color_list)))
    viz_csv(rmn_traj, rmn_descs, color_list,
            min_length=20, max_viz=True,
            fig_dir=save_path, smallest_shift=1)
