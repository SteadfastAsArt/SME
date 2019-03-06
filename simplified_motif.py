import time
import sys
import graph_utils as gu
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import random
from gensim.models import Word2Vec
from itertools import islice
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

tsne = TSNE(learning_rate=100)
pca = PCA(n_components=5)

__author__ = "Andrian Lee, Chen, Yubo Tao"


class Connector(object):
    def __init__(self, _anchors=[]):
        self.anchors = _anchors
        self.spanners = []

    def append_spanners(self, _spanners):
        self.spanners.append(_spanners)

    def __del__(self):
        pass


def find_fans(G):
    _fans = defaultdict(list)
    for n in G.nodes():
        if G.out_degree(n) > 1:
            _leaves = []
            for nbr in G[n]:
                if G.out_degree(nbr) == 1:
                    _leaves.append(nbr)
            if len(_leaves) > 1:
                _fans[(n,)].extend(_leaves)

    return _fans


def find_triad(G):
    _triad = defaultdict(list)
    tmp = []

    for n in G.nodes():
        if G.out_degree(n) == 2 and ((G.out_degree(G[n][0]) == 2 and G.out_degree(G[n][1]) > 2) or
                                     (G.out_degree(G[n][1]) == 2 and G.out_degree(G[n][0]) > 2)):
            tmp.append(n)
    print(len(tmp))

    triple = []
    for t in tmp:
        triple.append(sorted([t, G[t][0], G[t][1]]))
    sorted(triple)

    idx = 0
    while idx < len(triple)-1:
        if triple[idx] == triple[idx+1]:
            t = triple[idx]
            if G.out_degree(t[0]) > 2:
                _triad[(t[0],)] = [t[1], t[2]]
            elif G.out_degree(t[1]) > 2:
                _triad[(t[1],)] = [t[0], t[2]]
            else:
                _triad[(t[2],)] = [t[0], t[1]]
            idx += 2
        else:
            idx += 1


    # for idx, t in enumerate(tmp):
    #     for tt in tmp:
    #         if t != tt:
    #             if sorted([t, G[t][0], G[t][1]]) == sorted([tt, G[tt][0], G[tt][1]]):
    #                 if G[t][0] == G[tt][0] or G[t][0] == G[tt][1]:
    #                     _triad[(G[t][0],)] = [t, tt]
    #                 elif G[t][1] == G[tt][0] or G[t][1] == G[tt][1]:
    #                     _triad[(G[t][1],)] = [t, tt]

    return _triad


def find_connectors(G, D=2):
    _conn = defaultdict(list)
    for n in G.nodes():
        # D-Connector
        if G.out_degree(n) == D:
            _conn[tuple(sorted(G[n]))].append(n)

    for k in list(_conn.keys()):
        if len(_conn[k]) < 2:
            _conn.pop(k)

    return _conn


def collapse(G, rm_dict):
    nextid = max(G.nodes()) + 1
    transfer_dict = defaultdict(list)
    """
        rm_dict: { anchors: spanners, ..., }; from `find` procedure
        anchors: tuple
        spanners: list
    """
    for t in rm_dict.keys():
        transfer_dict[nextid] = rm_dict[t]
        # s: single spanner
        for s in rm_dict[t]:
            # remove spanner from graph
            if s in G.keys():
                G.pop(s)
                # x: single anchor
            for x in t:
                if s in G[x]:
                    # remove spanner from anchor
                    G[x].remove(s)
        # add the new substitute to graph
        for x in t:
            G[x].append(nextid)
            G[nextid].append(x)
        # renew the next id
        nextid += 1

    return transfer_dict


def training(G, _filepath, o=1, number_walks=10, walk_length=80, representation_size=64, window_size=5, max_memory_data_size = 6e8):
    data_size = number_walks * walk_length
    print("Data size (walks*length): {}".format(data_size))
    output = _filepath + G.name
    # if data_size < max_memory_data_size:
    if True:
        print("Walking...")
        time_start = time.time()
        walks = gu.build_deepwalk_corpus(G, num_paths=number_walks, path_length=walk_length,
                                         alpha=0, rand=random.Random(0))
        time_end = time.time()
        print('Walking time cost:', time_end - time_start)

        print("Training...")
        time_start = time.time()
        model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1, workers=5)
        time_end = time.time()
        print('Training vectors time cost:', time_end - time_start)
    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
        print("Walking...")
    if o == 1:
        model.wv.save_word2vec_format(output + '.emb')
    else:
        model.wv.save_word2vec_format(output + '0.emb')

    return time_end - time_start, 1


def main():

    _filepath = '../dataset/arenas-email/out.arenas-email'
    _filepath = '../dataset/opsahl-usairport/out.opsahl-usairport'
    _filepath = '../dataset/ca-AstroPh/out.ca-AstroPh'
    _filepath = 'D:/NRL/dataset/KONECT/brunson_revolution/out.brunson_revolution_revolution'

    G0 = gu.load_edgelist(_filepath)
    G_after = gu.load_edgelist(_filepath)

    print('*' * 40)
    print("Original graph:")
    G0_des = G0.statistic_description()

    print('*'*40)
    # print('Start detecting...')
    time_start = time.time()

    _fans = find_fans(G0)
    _conn2 = find_connectors(G0, D=2)
    _conn3 = find_connectors(G0, D=3)
    _triads = find_triad(G0)

    time_end = time.time()
    print('Detecting time cost:', time_end - time_start)

    print("number of fans: ", len(_fans))
    print("average of fan size: ", np.mean([len(_fans[k]) for k in _fans.keys()]))
    print("number of 2-connectors: ", len(_conn2))
    print("number of 3-connectors: ", len(_conn3))
    print("number of triads: ", len(_triads))

    print('*'*40)
    # print('Start transforming...')
    time_start = time.time()
    transfer_fans = collapse(G_after, _fans)
    transfer_conn2 = collapse(G_after, _conn2)
    transfer_conn3 = collapse(G_after, _conn3)
    transfer_triads = collapse(G_after, _triads)
    time_end = time.time()
    # print('End transforming...')
    print('Transforming time cost:', time_end - time_start)

    print('*' * 40)

    print("Transformed graph:")
    G_after_des = G_after.statistic_description()

    print('*' * 40)

    print("degree of reduction: ")
    print("nodes: ", round((G0_des[0] - G_after_des[0])/G0_des[0]*100, 2), " %")
    print("edges: ", round((G0_des[1] - G_after_des[1]) / G0_des[1]*100, 2), " %")

    print('*' * 40)

    G0_time = training(G0, o=0)
    G_after_time = training(G_after)

    print('*' * 40)

    print("Time reduction: ")
    print("actual: ", G0_time - G_after_time)
    print("portion: ", round((G0_time - G_after_time) / G0_time*100, 2), " %")

    print('*' * 40)
    # 3 layouts
    # gu.draw(gu.GraphImpl2nx(G0))
    # gu.draw(gu.GraphImpl2nx(G_after))


def test():
    Gt = gu.GraphImpl()
    """
    Gt[1] = [2, 3]
    Gt[2] = [1, 3]
    Gt[3] = [1, 2, 4]
    Gt[4] = [3, 5, 6]
    Gt[5] = [4, 6]
    Gt[6] = [4, 5]

    _triads = find_triad(Gt)
    print(_triads)
    transfer_triads = collapse(Gt, _triads)
    print(transfer_triads)
    gu.draw(gu.GraphImpl2nx(Gt))
    ------
    g = nx.Graph()
    pos = [(1, 1), (1, 2), (2, 2), (3, 4)]
    g.add_edges_from([(0, 1), (0, 2), (2, 3)])
    nx.draw(g, pos=pos, with_labels=True)

    plt.xlim(0, 5)  # 设置首界面X轴坐标范围
    plt.ylim(0, 5)  # 设置首界面Y轴坐标范围

    plt.show()
    """


if __name__ == '__main__':
    sys.exit(main())
    # sys.exit(test())
