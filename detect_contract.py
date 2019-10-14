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
import misc_utils as mu

tsne = TSNE(learning_rate=100)
pca = PCA(n_components=5)

__author__ = "Andrian Lee, Chen, Yubo Tao"


class AS(object):
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


def detect(G, _path):
    print("Start detecting...")
    start_t = time.time()
    _fans = find_fans(G)
    _conn2 = find_connectors(G, D=2)
    print("connectors finished")
    _triads = find_triad(G)
    end_t = time.time()
    time_delta = round(end_t - start_t, 3)
    print("End detecting...")

    det_out = [_fans, _conn2, _triads, time_delta]

    mu.serialization(_fans, _path+'/det_fans.'+G.name)
    mu.serialization(_conn2, _path + '/det_conn2.' + G.name)
    mu.serialization(_triads, _path + '/det_triads.' + G.name)

    return det_out


def collapse(G, rm_dict):
    transfer_dict = defaultdict(list)
    """
        rm_dict: { anchors: spanners, ..., }; from `find` procedure
        anchors: tuple
        spanners: list
        ---
        transfer_dict: { new_node: spanners }
    """
    nextid = max(G.nodes()) + 1

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


def transform(G, motif_list, _path):
    print("Start transforming...")
    start_t = time.time()
    transfer_fans = collapse(G, motif_list[0])
    transfer_conn2 = collapse(G, motif_list[1])
    transfer_triads = collapse(G, motif_list[2])
    end_t = time.time()
    time_delta = round(end_t - start_t, 3)
    print("End transforming...")

    trans_out = [transfer_fans, transfer_conn2, transfer_triads, time_delta]

    gu.GraphImpl2disk(G, _path+'/out.new.'+G.name)

    mu.serialization(transfer_fans, _path + '/trans_fans.' + G.name)
    mu.serialization(transfer_conn2, _path + '/trans_conn2.' + G.name)
    mu.serialization(transfer_triads, _path + '/trans_triads.' + G.name)

    return trans_out
