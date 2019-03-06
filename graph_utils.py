from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, Iterable
from six import iterkeys
import random
import networkx as nx
import pandas as pd
from time import time
from itertools import islice
import training_batch as tb
from six.moves import zip_longest
from multiprocessing import cpu_count
from os import path
__author__ = "Andrian Lee, Chen, Yubo Tao"


class GraphImpl(defaultdict):
    """Efficient basic implementation of network `Graph` Undirected graphs with self loops"""

    def __init__(self, _name=None):
        super(GraphImpl, self).__init__(list)
        self.name = _name

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def out_degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def stat_degree(self, _degree):
        cnt = 0
        for k in self.values():
            if len(k) == _degree:
                cnt += 1
        return cnt / len(self.keys())

    def statistic_description(self):
        print("nodes: ", self.number_of_nodes())
        print("edges: ", int(self.number_of_edges()))
        print("degree-1: ", self.stat_degree(1))
        print("degree-2: ", self.stat_degree(2))
        return (self.number_of_nodes(), int(self.number_of_edges()), self.stat_degree(1), self.stat_degree(2),
                self.stat_degree(3))

    def order(self):
        """Returns the number of nodes in the graph"""
        return len(self)

    def number_of_edges(self):
        """Returns the number of edges in the graph"""
        return sum([self.out_degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        """Returns the number of nodes in the graph"""
        return self.order()

    def make_undirected(self):
        for v in self.keys():
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        self.make_consistent()
        return self

    def make_consistent(self):
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):
        removed = 0

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1
        # print("remove ", removed, " keys...")
        # return self

    def subgraph(self, nodes={}):
        subgraph = GraphImpl()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
        -------------------------------------------------
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


def load_edgelist(_filepath, _name=None, undirected=True):
    f = open(_filepath)
    lines = f.readlines()
    f.close()

    G = GraphImpl(_name)
    for line in lines:
        line = line.strip()
        u, v = line.split()[:2]
        G[int(u)].append(int(v))
        if undirected:
            G[int(v)].append(int(u))
    G.make_consistent()
    return G


def load_adjlist():
    pass


def load_emb(_embpath):
    with open(_embpath) as f:
        lines = f.readlines()
    feature_set = []
    for line in islice(lines, 1, None):
        feature_set.append(line.split())

    return feature_set


def GraphImpl2nx(G):
    out = nx.Graph()
    for k in G.keys():
        for i in G[k]:
            out.add_edge(k, i)
    return out


def GraphImpl2disk(G, _path):
    f = open(_path, 'w+')
    for k in G.keys():
        for i in G[k]:
            f.writelines(str(k)+" "+str(i)+'\n')
    f.close()


def count_lines(f):
  if path.isfile(f):
    num_lines = sum(1 for line in open(f))
    return num_lines
  else:
    return 0


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    walks = []
    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

    return walks


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                               rand=random.Random(0)):
    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def _write_walks_to_disk(args):
    num_paths, path_length, alpha, rand, f = args
    G = __current_graph
    t_0 = time()
    with open(f, 'w') as fout:
        for walk in build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length,
                                               alpha=alpha, rand=rand):
            fout.write(u"{}\n".format(u" ".join(v for v in walk)))
    tb.logging("Generated new file {}, it took {} seconds".format(f, time() - t_0))
    return f


def write_walks_to_disk(G, filebase, num_paths, path_length, alpha=0, rand=random.Random(0), num_workers=cpu_count(),
                        always_rebuild=True):
    global __current_graph
    __current_graph = G
    files_list = ["{}.{}".format(filebase, str(x)) for x in list(range(num_paths))]
    expected_size = len(G)
    args_list = []
    files = []

    if num_paths <= num_workers:
        paths_per_worker = [1 for x in range(num_paths)]
    else:
        paths_per_worker = [len(list(filter(lambda z: z!= None, [y for y in x])))
                            for x in grouper(int(num_paths / num_workers)+1, range(1, num_paths+1))]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):
            if always_rebuild or size != (ppw*expected_size):
                args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2**31)), file_))
            else:
                files.append(file_)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file_ in executor.map(_write_walks_to_disk, args_list):
            files.append(file_)

    return files

