import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import sys
from sklearn.linear_model import LogisticRegression
import simplified_motif as sm
import graph_utils as gu
import numpy as np
from time import ctime
import pickle as pk
import csv
import os
import observation as ob
import itertools
import random
import math
import performance_analysis as pa
from collections import defaultdict


def uniform_sphere_sampling(dim, N, r):
    vec = np.random.randn(dim, N)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= r
    return vec


def release_specific(_motif, trans_motif, emb, _dir, radius2, flag_triad=False, flag_conn=False):
    """ Release new node back to original
    :param _motif:
    :param trans_motif:
    :param g:
    :param emb: pre-trained emb., not emb0
    :param _dir:
    :param flag_triad:
    :param flag_conn:
    :return:
    """

    _map = defaultdict(list)  # _map: new node -> anchors:list
    for k in _motif.keys():
        for kk in trans_motif.keys():
            if _motif[k] == trans_motif[kk]:
                for _k in k:  # tuple k -> _k
                    _map[kk].append(_k)  # anchors:list

    out = []
    with open(_dir + 'test1.emb', 'a+') as f:  # todo: test.emb
        central_np = ob.get_central(emb)
        for k in emb.keys():
            if k in trans_motif.keys():  # k -> new node
                # suitable for spanners with interactions
                if flag_triad:  # different spanners' rules towards triads
                    _r = ob.euclidean_dist(emb[_map[k][-1]], emb[k])
                    # _r = 0.0
                elif flag_conn:
                    # _r = np.mean([ob.euclidean_dist(emb[kk], emb[k]) for kk in _map[k]])
                    # _r = ob.euclidean_dist(emb[_map[k][-1]], emb[k])
                    _r = radius2 * 0.8
                else:  # for fans
                    _r = radius2

                if not flag_conn:  # different center rules towards connectors
                    center = _map[k][0]
                else:
                    center = k

                for item in trans_motif[k]:
                    cnt = 0
                    v1 = np.array(emb[center]) - central_np  # central -> anchor
                    while True:
                        # todo: dimension 128 is hyper_parameter; hyper-sphere surface sampling
                        vec_list = uniform_sphere_sampling(128, 1, _r).transpose()
                        if not flag_conn and _r != 0.0 and ob.cosine(v1, vec_list[0]) < 0:
                            cnt += 1
                            continue
                        else:
                            break
                    _vec = []
                    for idx, i in enumerate(vec_list[0]):
                        # centered at anchor/(new_node)
                        _vec.append(round(emb[center][idx] + i, 8))
                    f.write(str(item))
                    for x in _vec:
                        f.write(" " + str(x))
                    f.write('\n')

                out.append(k)

    return out


def get_r2(g, emb):
    # control the sampling size
    if g.number_of_edges() < 3e4:
        _rate = 0
    else:
        _rate = int(3e4)
    x1, l1 = pa.sample_positive(g, emb, _rate)
    x2, l2 = pa.sample_negative(g, emb, len(x1))
    _X = (x1 + x2)
    X = np.array(_X).reshape(-1, 1)
    _label = (l1 + l2)
    LogR = LogisticRegression(C=1e5, )
    print('Start fit:', ctime())
    clf_logR = LogR.fit(X, _label)
    print('End fit:', ctime())
    # radius2: suitable for spanners with no interactions
    # radius2 = sum(x1+x2) / len(x1+x2)
    radius2 = LogR.intercept_ / LogR.coef_
    print("radius2: {}".format(radius2))
    return radius2


def release(_dir, _name, g_fil):
    fan = ob.loadpy2mem(_dir + 'det_fans.' + _name)
    conn2 = ob.loadpy2mem(_dir + 'det_conn2.' + _name)
    traid = ob.loadpy2mem(_dir + 'det_triads.' + _name)

    trans_fan = ob.loadpy2mem(_dir + 'trans_fans.' + _name)
    trans_conn2 = ob.loadpy2mem(_dir + 'trans_conn2.' + _name)
    trans_triad = ob.loadpy2mem(_dir + 'trans_triads.' + _name)

    emb  = gu.load_emb2(_dir + 'n2v.' + _name + '.emb')
    emb0 = gu.load_emb2(_dir + 'n2v.' + _name + '.emb0')
    print('original emb. size: {}'.format(len(emb0)))

    # g0 = gu.load_edgelist(_dir + g_fil, _name)
    g = gu.load_edgelist(_dir + g_fil, _name=_name)

    radius2 = get_r2(g, emb)
    a = release_specific(fan, trans_fan, emb, _dir, radius2)
    b = release_specific(conn2, trans_conn2, emb, _dir, radius2, flag_conn=True)
    c = release_specific(traid, trans_triad, emb, _dir, radius2, flag_triad=True)
    rr = a+b+c
    todo = []
    for k in emb.keys():
        if k not in rr:
            todo.append(k)

    # take care of those nodes not in motifs
    with open(_dir + 'test1.emb', 'a+') as f:  # todo: name rules _final postfix
        for t in todo:
            f.write(str(t))
            for item in emb[t]:
                f.write(" " + str(item))
            f.write('\n')

    with open(_dir + 'test1.emb', 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(str(len(emb0)) + " 128\n"+content)

    test = gu.load_emb2(_dir + 'test1.emb')
    print('End:', ctime())
    print('post-trained emb. size: {}'.format(len(test)))
    print('post-trained emb. dimension: {}'.format(len(test[1])))
    del test


if __name__ == '__main__':
    root = 'D:\\NRL\dataset\KONECT\\'
    _name = 'arenas-pgp'
    _dir = root + _name + '/'
    # g_file = 'out.new.arenas-pgp'
    g_file = 'out.new.arenas-pgp'
    release(_dir, _name, g_file)
