import csv
import gzip
import numpy as np
import pickle


def load_data(span_path, metadata_path):
    x = csv.DictReader(gzip.open(span_path, 'rt', encoding='utf8'))
    wmap, cmap, bmap = pickle.load(open(metadata_path, 'rb'))

    revwmap = dict((v, k) for (k, v) in wmap.items())

    span_dict = {}
    for row in x:
        text = row['Words'].split()
        key = '___'.join([row['Book'], row['Char 1'], row['Char 2']])
        if key not in span_dict:
            span_dict[key] = []
        span_dict[key].append([wmap[w] for w in text])

    new_dict = {}
    span_info = {}
    for key in span_dict:

        book, c1, c2 = key.split('___')
        span_info[key] = (book, c1, c2)

        spans = span_dict[key]
        max_len = max(map(len, spans))

        s = -1 * np.ones((len(spans), max_len)).astype('int32')
        for i, curr_span in enumerate(spans):
            s[i][:len(curr_span)] = curr_span

        new_dict[key] = s

    return new_dict, span_info, revwmap


def load_We():
    with open('data/glove.We', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        We = p.astype('float32')

    norm_We = We / np.linalg.norm(We, axis=1)[:, None]
    We = np.nan_to_num(norm_We)
    return We
