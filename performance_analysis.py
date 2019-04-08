import pickle as pk
from sklearn.linear_model import LogisticRegression
import observation as ob
import graph_utils as gu
from sklearn.metrics import roc_auc_score, roc_curve
import random
import numpy as np
from time import ctime


def sample_positive(g, emb, rate=0):
    X = []
    lable = []

    print('Start sample_positive', ctime())
    for k in g.keys():
        for i in g[k]:
            if k < i:  # avoid duplicated edges
                X.append(ob.euclidean_dist(emb[k], emb[i]))  # What kind of distance of long distance todo
                lable.append(1)
    if rate != 0:
        X = random.sample(X, rate)
        lable = lable[:rate]
    print('num_postive samples:', len(X))
    print('End sample_positive', ctime())
    return X, lable


def sample_negative(g, emb, rate):
    """ Rules on (Negative) Sampling
    ------
    + easy way: several times than positive samples randomly
    + core vertex set: each none edge vertex pairs to be 0
    + task specified: e.g. only focus on the none-edges near the actual edges

    ++ dataset spliting: hide some actual edges to be tested & also task specified
    """
    X = []
    lable = []
    num_v = len(g)
    num_e = int(num_v * (num_v - 1) / 2)

    interV = [0]

    """ Be careful about the matrix 0-index """
    for i in range(num_v-1, 0, -1):
        interV.append(interV[-1] + i)

    print('Start sample_negative', ctime())
    ran_sampl = random.sample(range(1, num_e+1), rate)  # edge indexed from 1

    gk = sorted(g.keys())
    for r in ran_sampl:
        i = 0
        for idx, inter in enumerate(interV):  # todo: may have chance to enhance
            if r <= inter:
                i = idx - 1
                break

        j = int(r - (2 * num_v - i - 3) * i / 2)
        if gk[j] not in g[gk[i]]:  # map each node_id to a continuous id_list
            X.append(ob.euclidean_dist(emb[gk[i]], emb[gk[j]]))
            lable.append(0)
    print('num_negative samples:', len(X))
    print('End sample_negative', ctime())
    return X, lable


def full_sample(g, emb, undirected=True):
    res = []
    label = []
    k = g.keys()
    for idi, i in enumerate(k):
        if idi % 1000 == 0:
            print('yes' + ctime())
        for idj, j in enumerate(k):
            if idi < idj:
                try:
                    res.append(ob.euclidean_dist(emb[i], emb[j]))
                    label.append(1 if j in g[i] else 0)
                except Exception as ex:
                    print(ex, end=' ')
                    print(i, j)
    print('num_full samples:', len(res))
    return res, label


def f1_score(lable_true, lable_pre):
    tp = tn = fp = fn = 0
    for i, j in zip(lable_true, lable_pre):
        if i == 1 and j == 1:
            tp += 1
        elif i == 1 and j == 0:
            fn += 1
        elif i == 0 and j == 0:
            tn += 1
        else:
            fp += 1
    recall = precision = specificity = -1
    if tp + fn != 0:
        recall = tp / (tp + fn)
    if tp + fp != 0:
        precision = tp / (tp + fp)
    if tn + fp != 0:
        specificity = tn / (tn + fp)
    print('tp:{}, tn:{}, fp:{}, fn:{}'.format(tp, tn, fp, fn))
    print('Recall: {} / {} ({})'.format(tp, tp + fn, recall))
    print('Precision: {} / {} ({})'.format(tp, tp + fp, precision))
    print('Specificity: {} / {} ({})'.format(tn, tn + fp, specificity))
    print('F1-score: {}'.format(2 * recall * precision / (recall + precision)))


def motif_performance_specific(_motif, clf, g, emb, _type):
    """ { anchors: spanners }
    3 types of relations:
    + anchors <-> spanners: edge
    + spanners <-> spanners: depends
    + spanners <-> outside world: no edge
    (while anchors will alway have edge with outside)
    :param _motif:
    :return:
    """
    # anchors <-> spanners: edge
    print('anchors <-> spanners: edge')
    l1 = []
    _x1 = []
    for k in _motif.keys():
        for kk in k:
            for item in _motif[k]:
                _x1.append(ob.euclidean_dist(emb[kk], emb[item]))
                l1.append(1)
    x1 = np.array(_x1).reshape(-1, 1)
    l1_pre = clf.predict(x1)
    f1_score(l1, l1_pre)

    # spanners <-> spanners: depends
    print('spanners <-> spanners: depends')
    l2 = []
    _x2 = []
    for k in _motif.keys():
        for idi, i in enumerate(_motif[k]):
            for idj, j in enumerate(_motif[k]):
                if idi < idj:
                    _x2.append(ob.euclidean_dist(emb[i], emb[j]))
                    l2.append(1 if _type == 'triad' else 0)
    x2 = np.array(_x2).reshape(-1, 1)
    l2_pre = clf.predict(x2)
    f1_score(l2, l2_pre)

    # spanners <-> outside world: no edge
    print('spanners <-> outside world: no edge')
    l3 = []
    _x3 = []
    for k in _motif.keys():
        for s in _motif[k]:  # s: spanner
            for n in g.keys():
                if n not in _motif.keys() and n not in _motif[k]:  # n: not in motif
                    _x3.append(ob.euclidean_dist(emb[s], emb[n]))
                    l3.append(0)
    x3 = np.array(_x3).reshape(-1, 1)
    l3_pre = clf.predict(x3)
    f1_score(l3, l3_pre)


def motif_performance_compare(dir, name, clf, g, emb, f1=1, f2=1, f3=1):
    """ To measure the performance of tasks on certain motifs
    against overall results. To be more specific, motifs are divided
    into the 4 types and have a overall result.
    :parameter
    :return:
    """
    fan = ob.loadpy2mem(dir + 'det_fans.' + name)
    conn2 = ob.loadpy2mem(dir + 'det_conn2.' + name)
    triad = ob.loadpy2mem(dir + 'det_triads.' + name)
    if f1 == 1:
        print('\n'+'-'*30+'fan')
        motif_performance_specific(fan, clf, g, emb, 'fan')
    if f2 == 1 and len(conn2) > 0:
        print('\n'+'-'*30+'conn2')
        motif_performance_specific(conn2, clf, g, emb, 'conn2')
    if f3 == 1 and len(triad) > 0:
        print('\n'+'-'*30+'triad')
        motif_performance_specific(triad, clf, g, emb, 'triad')


def main(ifFull=False, _original=False):
    root = 'D:\\NRL\dataset\KONECT\\'
    _name = 'arenas-pgp'
    _dir = root + _name + '/'

    g_file = 'out.arenas-pgp'
    if _original:
        emb_file = 'dblp-cite0.emb'
    else:
        emb_file = 'test1.emb'

    print('Start', ctime())
    print(_name)
    emb0 = gu.load_emb2(_dir + emb_file)
    g0 = gu.load_edgelist(_dir + g_file, _name=_name, )

    _X = []
    _label = []
    x1, l1 = sample_positive(g0, emb0)
    x2, l2 = sample_negative(g0, emb0, len(x1))
    _X += x1
    _label += l1
    _X += x2
    _label += l2
    X = np.array(_X).reshape(-1, 1)
    del _X, l1, l2

    LogR = LogisticRegression(C=1e5, solver='liblinear')
    print('Start fit:', ctime())
    clf_logR = LogR.fit(X, _label)
    print('End fit:', ctime())
    print("Logistic Regression params: {}, {}".format(LogR.coef_, LogR.intercept_))
    print("threshold: {}".format(LogR.intercept_ / LogR.coef_))

    # Overall results
    print('*' * 50)
    if ifFull:
        print('Overall:')
        _T, lable_all = full_sample(g0, emb0)
        T = np.array(_T).reshape(-1, 1)
        del _T

        print('Start predict:', ctime())
        pre_lable = LogR.predict(T)  # all of V * (V - 1) / 2 samples
        pre_prob = LogR.predict_proba(T)
        print('End predict:', ctime())

        f1_score(lable_all, pre_lable)
        auc1 = roc_auc_score(lable_all, pre_lable)
        print('auc-lable: ', auc1)
        auc2 = roc_auc_score(lable_all, pre_prob[:, 1])
        print('auc-prob: ', auc2)
    else:
        print('Motif specific:')
        motif_performance_compare(_dir, _name, LogR, g0, emb0, f1=0, f2=1, f3=1)

    print('End', ctime())


if __name__ == '__main__':
    main()
