import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import sys
from sklearn.linear_model import LogisticRegression
import detect_contract as sm
import graph_utils as gu
import numpy as np
import time
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


def release_specific(_motif, trans_motif, emb, final_emb, radius2, dim,  flag_triad=False, flag_conn=False):
    """ Release new node back to original
    :param _motif:
    :param trans_motif:
    :param g:
    :param emb: pre-trained emb. to be released, not the original emb0
    :param final_emb:
    :param radius2:
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
    with open(final_emb, 'a+') as f:
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
                    # _r = radius2 * 0.8
                    _r = radius2
                else:  # for fans
                    _r = radius2

                if not flag_conn:  # different center rules towards connectors
                    center = _map[k][0]
                else:
                    center = k

                for item in trans_motif[k]:
                    cnt = 0
                    v1 = np.array(emb[center]) - central_np  # central -> anchor
                    # while True:
                    #     # todo: dimension 128 is hyper_parameter that could be toned; hyper-sphere surface sampling
                    #     # todo: THIS is slow !!!!
                    #     vec_list = uniform_sphere_sampling(dim, 1, _r).transpose()
                    #     # if not flag_conn and _r != 0.0 and ob.cosine(v1, vec_list[0]) < -1/2:
                    #     if not flag_conn and _r != 0.0:
                    #         cnt += 1
                    #         continue
                    #     else:
                    #         break
                    vec_list = uniform_sphere_sampling(dim, 1, _r).transpose()
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
    # radius2: suitable for spanners with no interactions
    # control the sampling size
    if g.number_of_edges() < 1e4:  # todo: from 3e4 -> 1e4
        _rate = int(len(g)/3)
    else:
        _rate = int(len(g)/20)
    x1, l1 = pa.sample_positive(g, emb, _rate)
    x2, l2 = pa.sample_negative(g, emb, len(x1))
    _X = (x1 + x2)
    X = np.array(_X).reshape(-1, 1)
    _label = (l1 + l2)
    LogR = LogisticRegression(C=1e5, )
    print('Start fit:', time.ctime())
    clf_logR = LogR.fit(X, _label)
    print('End fit:', time.ctime())

    # radius2 = sum(x1+x2) / len(x1+x2)
    radius2 = LogR.intercept_ / LogR.coef_
    print("radius2: {}".format(radius2))
    return radius2


def release(_dir, _name, g_fil, method, dim):
    print('Release start: {}', time.ctime())
    start = time.time()

    fan = ob.loadpy2mem(_dir + 'det_fans.' + _name)
    conn2 = ob.loadpy2mem(_dir + 'det_conn2.' + _name)
    triad = ob.loadpy2mem(_dir + 'det_triads.' + _name)

    trans_fan = ob.loadpy2mem(_dir + 'trans_fans.' + _name)
    trans_conn2 = ob.loadpy2mem(_dir + 'trans_conn2.' + _name)
    trans_triad = ob.loadpy2mem(_dir + 'trans_triads.' + _name)

    emb = gu.load_emb2(_dir + _name + '.' + method + '.emb')
    emb0 = gu.load_emb2(_dir + _name + '0.' + method + '.emb')
    print('original emb. size: {}'.format(len(emb0)))
    # manipulate on the new_graph
    g = gu.load_edgelist(_dir + g_fil, _name=_name)

    start_r2 = time.time()
    radius2 = get_r2(g, emb)
    end_r2 = time.time()
    print('R2 time:', end_r2 - start_r2)

    final_emb = _dir + _name + '1.' + method + '.emb'
    a = release_specific(fan, trans_fan, emb, final_emb, radius2, dim)
    b = release_specific(conn2, trans_conn2, emb, final_emb, radius2, dim, flag_conn=True)
    c = release_specific(triad, trans_triad, emb, final_emb, radius2, dim, flag_triad=True)
    rr = a+b+c
    todo = []

    for k in emb.keys():
        if k not in rr:
            todo.append(k)

    start = time.time()
    # take care of those nodes not in motifs
    with open(final_emb, 'a+') as f:
        for t in todo:
            f.write(str(t))
            for item in emb[t]:
                f.write(" " + str(item))
            f.write('\n')
    end = time.time()
    print(end - start)
    with open(final_emb, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(str(len(emb0)) + " 128\n"+content)

    end = time.time()

    print('Release End, total time cost: {}', end-start)

    test = gu.load_emb2(final_emb)
    print('post-trained emb. size: {}'.format(len(test)))
    print('post-trained emb. dimension: {}'.format(len(test[1])))
    del test


if __name__ == '__main__':
    root = './dataset/_social/'
    _name = 'arenas-pgp'
    g_file = 'out.new.arenas-pgp'  # new graph
    method = 'dw'  # enum: {gf, dw, n2v, line, sage}
    dim = 128

    _dir = root + _name + '/'
    release(_dir, _name, g_file, method, dim, )
