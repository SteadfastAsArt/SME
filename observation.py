import graph_utils as gu
import pickle as pk
import math
import numpy as np


def euclidean_dist(u, v):
    _delta = 0.0
    for i, j in zip(u, v):
        _delta += (i-j)**2
    res = math.sqrt(_delta)
    return res


def cosine(u, v):
    """ cos(u, v) = u * v / |u|*|v|
    :param u: vector1
    :param v: vector2
    :return: cos(u, v)
    """
    len_u = math.sqrt(sum([i*i for i in u]))
    len_v = math.sqrt(sum([i*i for i in v]))
    u_v = sum([i*j for i, j in zip(u, v)])
    res = u_v / (len_u * len_v)
    return res


def mean_dis(g, emb, ):
    """ uniformly sampling equal-quantity +/- edges
    :param g:
    :param emb:
    :return:
    """
    num_edge = 0
    sum_dist = 0.0
    for k in g.keys():
        for i in g[k]:
            sum_dist += euclidean_dist(emb[k], emb[i])
            num_edge += 1
    res = sum_dist / num_edge
    return res


def loadpy2mem(_path):
    obj = pk.load(open(_path, 'rb'))
    return obj


def get_central(emb_dict):
    tmp = []
    for v in emb_dict.values():
        tmp.append(v)
    array = np.array(tmp)
    _res = np.mean(array, axis=0)
    # res = _res.tolist()
    return _res


def anchor2newnode(_motif, trans_motif):
    _map = dict()  # _map: new node -> anchors
    for k in _motif.keys():
        for kk in trans_motif.keys():
            if _motif[k] == trans_motif[kk]:
                for _k in k:  # tuple k -> _k
                    _map[_k] = kk  # truncate: last come, stay; may not suitable for conn2
    return _map


def observe(motif, emb, meand=0):
    # anchors <-> spanners
    lesser = 0
    tot = 0
    for k in motif.keys():
        for kk in k:
            for i in motif[k]:
                tot += 1
                if meand > euclidean_dist(emb[kk], emb[i]):
                    lesser += 1

    print('anchors <-> spanners: {} / {}, {}'.format(lesser, tot, round(lesser/tot, 4)))

    # spanners <-> spanners
    lesser = 0
    tot = 0
    for k in motif.keys():
        for i in motif[k]:
            for ii in motif[k]:
                if i != ii:
                    tot += 1
                    if meand > euclidean_dist(emb[i], emb[ii]):
                        lesser += 1

    print('spanners <-> spanners: {} / {}, {}'.format(lesser, tot, round(lesser / tot, 4)))


def main():
    _name_list = []
    with open('todolist2.txt') as f:
        _name_list = [x.strip() for x in f.readlines()]
    _root = 'D:\\NRL\dataset\KONECT\\'
    for _name in _name_list:
        print('*' * 40)
        print(_name)

        dir = _root + _name + '/'
        _file = 'out.'+_name
        _file1 = _name+'0.emb'
        _file2 = 'det_fans.'+_name
        _file3 = 'det_conn2.'+_name
        # _file4 = 'det_conn3.'+_name
        _file5 = 'det_triads.'+_name
        g0 = gu.load_edgelist(dir+_file, _name)
        emb0 = gu.load_emb2(dir+_file1)  # d = 64
        meand = mean_dis(g0, emb0)
        print(meand)

        # fan = loadpy2mem(dir+_file2)
        # conn2 = loadpy2mem(dir+_file3)
        # # conn3 = loadpy2mem(dir + _file4)
        # triad = loadpy2mem(dir+_file5)
        # print(round(meand, 6))
        # observe(fan, emb0, meand)
        # observe(conn2, emb0, meand)
        # # observe(conn3, emb0, meand)
        # observe(triad, emb0, meand)

        print('*' * 40)


if __name__ == '__main__':
    main()

