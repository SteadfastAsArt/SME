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


def training(G, _filepath, o=1, number_walks=10, walk_length=80, representation_size=128, window_size=5, max_memory_data_size = 6e8):
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
