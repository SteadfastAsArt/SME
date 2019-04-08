import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import graph_utils as gu
import pickle as pk
import os
from time import ctime
from itertools import islice
tsne = TSNE(learning_rate=100)
pca = PCA(n_components=5)

# plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300


def draw_test(G):
    nx.draw(G, pos=nx.spring_layout(G), with_labels=True)
    plt.title("spring_layout")
    plt.show()

    nx.draw(G, pos=nx.kamada_kawai_layout(G), with_labels=True)
    plt.title("kamada_kawai_layout")
    plt.show()

    nx.draw(G, pos=nx.fruchterman_reingold_layout(G), with_labels=True)
    plt.title("fruchterman_reingold_layout")
    plt.show()


def emb2coor(g, feature_set, method='pca'):

    raw_data = pd.DataFrame(feature_set)
    data = raw_data.iloc[:, 1:]

    res = []
    if method == 'pca':
        res = pca.fit_transform(data)
    elif method == 'tsne':
        res = tsne.fit_transform(data)

    coor_dict = dict()
    for i, j in zip(feature_set, res):
        coor_dict[int(i[0])] = (j[0], j[1])
    # nx.draw(gu.GraphImpl2nx(g), pos=coor_dict, with_labels=True)
    # plt.show()
    return coor_dict


def matplotlib_draw(g, node_dict, coor_dict, trans_dict=None, o=1, type='fans', ):
    for k in coor_dict.keys():  # each node->purple
        plt.scatter(coor_dict[k][0], coor_dict[k][1], c='purple', s=10)

    for k in node_dict.keys():
        for a in k:  # i: each anchor->red
            plt.scatter(coor_dict[a][0], coor_dict[a][1], c='red', s=10)

    new_dict = dict()
    ann_list = []
    if trans_dict is not None:
        tmp_dict = dict()
        for kt in trans_dict.keys():
            tmp_dict[tuple(trans_dict[kt])] = [kt]
        for k in node_dict.keys():
            new_dict[k] = tmp_dict[tuple(node_dict[k])]
    else:  # original graph
        new_dict = node_dict

    # s: each spanner->yellow
    for k in new_dict.keys():
        for s in new_dict[k]:
            plt.scatter(coor_dict[s][0], coor_dict[s][1], c='yellow', s=3)

    # relevant edges
    for k in new_dict.keys():
        for a in k:
            for s in new_dict[k]:
                plt.plot([coor_dict[a][0], coor_dict[s][0]], [coor_dict[a][1], coor_dict[s][1]], lw=0.75, c='limegreen')

    print(new_dict)

    # # annotation
    # for k in new_dict.keys():
    #     ann_list.extend(new_dict[k])
    #     for i in k:
    #         ann_list.append(i)
    # for ann in ann_list:
    #     plt.plot([coor_dict[ann][0], coor_dict[ann][0]+0.013], [coor_dict[ann][1], coor_dict[ann][1]+0.013],
    #              lw=0.75, c='black')
    #     plt.annotate(ann, xy=(coor_dict[ann][0], coor_dict[ann][1]),
    #                  xytext=(coor_dict[ann][0] + 0.015, coor_dict[ann][1] + 0.015), fontsize=3)

    root = './pic_tsne/'
    if o == 1:
        tit = g.name + '-' + type
        plt.title(tit)
        plt.savefig(root+tit+'.jpg', dpi=300)
    else:
        tit = g.name + '0-' + type
        plt.title(tit)
        plt.savefig(root+tit+'.jpg', dpi=300)
    plt.close()
    # plt.show()


def vis_pipeline(method):
    root = 'D:/NRL/dataset/KONECT/'
    name_list = []
    _name_list = []
    ori_graph_path = []
    # for n in os.listdir(root):
    #     _name_list.append(n)
    with open('todolist2.txt') as f:
        _name_list = [x.strip() for x in f.readlines()]

    for _name in _name_list:
        _filepath = []
        for dirpath, dirnames, filenames in os.walk(root + _name):
            _filepath = [filepath for filepath in filenames if filepath[0:3] == 'out' and len(filepath.split('.')) == 2]
        _filename = "/" + _filepath[0]

        # if _name not in done_list and os.path.getsize(root + _name + '/' + _filename) / (1024 * 1024) <= 10.0:
        ori_graph_path.append(_filename)
        name_list.append(_name)

    for _name, _op in zip(name_list, ori_graph_path):
        print('To draw graph: ', _name)
        print(root + _name + _op)
        print(ctime())
        g = gu.load_edgelist(root + _name + _op, _name=_name)  # todo
        gg = gu.load_edgelist(root + _name + '/out.new.' + _name, _name=_name)
        print('*'*50)
        print('graph g : ', g)
        print('graph gg: ', gg)

        motif_list = ['fans', 'conn2', 'conn3', 'triads']

        try:
            feature_set0 = gu.load_emb(root + _name + '/' + _name + '0.emb')
            coor_dict0 = emb2coor(g, feature_set0, method=method)

            feature_set = gu.load_emb(root + _name + '/' + _name + '.emb')
            coor_dict = emb2coor(gg, feature_set, method=method)

            det_list = []
            trans_list = []
            for m in motif_list:
                try:
                    with open(root + _name + '/det_' + m + '.' + _name, 'rb') as f:
                        det_list.append(pk.load(f))
                    with open(root + _name + '/trans_' + m + '.' + _name, 'rb') as f:
                        trans_list.append(pk.load(f))
                except Exception as e:
                    print(repr(e))

            for m, d in zip(motif_list, det_list):
                matplotlib_draw(g, d, coor_dict0, o=0, type=m, )

            for m, d, t in zip(motif_list, det_list, trans_list):
                matplotlib_draw(gg, d, coor_dict, trans_dict=t, type=m, )

        except Exception as e:
            print(repr(e))

        print('*' * 50)


def main():
    root = 'D:/NRL/dataset/KONECT/'
    # _name = 'maayan-pdzbase'
    _name = 'contact'

    g = gu.load_edgelist(root + _name + '/out.'+_name, _name=_name)
    gg = gu.load_edgelist(root + _name + '/out.new.'+_name, _name=_name)

    print('graph g : ', g)
    print('graph gg: ', gg)

    motif_list = ['fans', 'conn2', 'conn3', 'triads']
    det_list = []
    trans_list = []
    for m in motif_list:
        with open(root+_name+'/det_'+m+'.'+_name, 'rb') as f:
            det_list.append(pk.load(f))
        with open(root+_name+'/trans_'+m+'.'+_name, 'rb') as f:
            trans_list.append(pk.load(f))

    feature_set0 = gu.load_emb(root+_name+'/'+_name+'0.emb')
    coor_dict0 = emb2coor(g, feature_set0)
    print('finish1')
    feature_set = gu.load_emb(root+_name+'/'+_name+'.emb')
    coor_dict = emb2coor(gg, feature_set)
    print('finish2')

    for m, d in zip(motif_list, det_list):
        matplotlib_draw(g, d, coor_dict0, o=0, type=m, )
    for m, d, t in zip(motif_list, det_list, trans_list):
        matplotlib_draw(gg, d, coor_dict, trans_dict=t, type=m, )


if __name__ == '__main__':
    # main()
    vis_pipeline('tsne')
