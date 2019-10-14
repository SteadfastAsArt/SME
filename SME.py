import detect_contract as dc
import graph_utils as gu
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import training_method as tm
import time
import numpy as np
import csv


def graph_load(_file, name, _type=None):
    """  """
    G = gu.load_edgelist(_file, _name=name, _type=_type)
    return G


def detect_contract(G0, G_after, root, name):
    path = root + name
    det_res = dc.detect(G0, path)
    # stats ...
    num_fans = len(det_res[0])
    avg_fan_size = round(float(np.mean([len(det_res[0][k]) for k in det_res[0].keys()])), 2)
    num_conn2 = len(det_res[1])
    num_traids = len(det_res[2])
    trans_res = dc.transform(G_after, det_res, path)

    return [num_fans, avg_fan_size, num_conn2, num_traids, det_res[-1], trans_res[-1]]


def training(G0, G_after, root, name):
    time_G0 = round(tm.deepwalk(G0, root + name + '/', o=0), 3)
    time.sleep(3)
    time_Gaf = round(tm.deepwalk(G_after, root + name + '/'), 3)
    time.sleep(3)
    return [time_G0, time_Gaf, name]


def release():
    pass


def output_stats(stats_tr, stats_dc):
    # 'graph_name', 'train_t_0', 'train_t_af', 't_d','t_c', 't_release', 'train_t_reduc'
    out_report = [stats_tr[2], stats_tr[0], stats_tr[1], stats_dc[-2], stats_dc[-1], '', '', ]
    with open('SME_report_final.csv', 'a', newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(out_report)


def main():
    parser = ArgumentParser("SME", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--root', default='./dataset/_social/', )
    parser.add_argument('--name', default='arenas-pgp', )
    parser.add_argument('--gfile0', default='out.arenas-pgp', )
    parser.add_argument('--gtype', default=None, )
    parser.add_argument('--method', default='deepwalk', )
    # parser.add_argument('--gfilenew', default='', )
    # parser.add_argument('--emb0', default='', )  # graphname(0).method.emb
    # parser.add_argument('--emb', default='', )
    # parser.add_argument('--emb1', default='', )
    args = parser.parse_args()

    root = args.root
    name = args.name
    gfile0 = args.gfile0
    gtype = args.gtype
    # gfilenew = args.gfilenew
    # emb0 = args.emb0
    # emb = args.emb
    # emb1 = args.emb1

    print("Start loading Graph: {}...".format(name))
    G0 = graph_load(root+name+'/'+gfile0, name, gtype)
    G_after = graph_load(root+name+'/'+gfile0, name, gtype)
    print("Loading finished!")

    stats_dc = detect_contract(G0, G_after, root, name)
    print('num_node      : %d' % G0.number_of_nodes())
    print('num_node after: %d' % G_after.number_of_nodes())
    print('num_edge      : %d' % G0.number_of_edges())
    print('num_edge after: %d' % G_after.number_of_edges())
    print('hub        : %d, %f' % (stats_dc[0], stats_dc[1]))
    print('2-connector: %d' % stats_dc[2])
    print('3-clique   : %d' % stats_dc[3])

    if args.method == 'deepwalk':  # integrated with deepwalk's training method
        stats_tr = training(G0, G_after, root, name)
        output_stats(stats_tr, stats_dc)


def batch():
    pass


if __name__ == '__main__':
    main()
