import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import sys
import simplified_motif as sm
import graph_utils as gu
import numpy as np
import time
import pickle as pk
import csv
import os


def serialization(obj, _path):
    with open(_path, 'wb') as f:
        pk.dump(obj, f)


def detect(G, _path):
    print("Start detecting...")
    start_t = time.time()
    _fans = sm.find_fans(G)
    _conn2 = sm.find_connectors(G, D=2)
    _conn3 = sm.find_connectors(G, D=3)
    print("connectors finished")
    _triads = sm.find_triad(G)
    end_t = time.time()
    time_delta = round(end_t - start_t, 3)
    print("End detecting...")

    det_out = [_fans, _conn2, _conn3, _triads, time_delta]

    serialization(_fans, _path+'/det_fans.'+G.name)
    serialization(_conn2, _path + '/det_conn2.' + G.name)
    serialization(_conn3, _path + '/det_conn3.' + G.name)
    serialization(_triads, _path + '/det_triads.' + G.name)

    return det_out


def transform(G, motif_list, _path):
    print("Start transforming...")
    start_t = time.time()
    transfer_fans = sm.collapse(G, motif_list[0])
    transfer_conn2 = sm.collapse(G, motif_list[1])
    transfer_conn3 = sm.collapse(G, motif_list[2])
    transfer_triads = sm.collapse(G, motif_list[3])
    end_t = time.time()
    time_delta = round(end_t - start_t, 3)
    print("End transforming...")

    trans_out = [transfer_fans, transfer_conn2, transfer_conn3, transfer_triads, time_delta]

    gu.GraphImpl2disk(G, _path+'/out.new.'+G.name)

    serialization(transfer_fans, _path + '/trans_fans.' + G.name)
    serialization(transfer_conn2, _path + '/trans_conn2.' + G.name)
    serialization(transfer_conn3, _path + '/trans_conn3.' + G.name)
    serialization(transfer_triads, _path + '/trans_triads.' + G.name)

    return trans_out


def logging(_info):
    with open('_log.txt', 'a') as f:
        f.writelines(_info+'\n')


def main():
    # root = 'D:/NRL/dataset/KONECT/'
    root = 'D:/NRL/dataset/bigdata/'

    name_list = []
    _name_list = []
    ori_graph_path = []

    with open('todolist.txt') as f:
        _name_list = [x.strip() for x in f.readlines()]

    for _name in _name_list:
        _filepath = []
        for dirpath, dirnames, filenames in os.walk(root + _name):
            _filepath = [filepath for filepath in filenames if filepath[0:3] == 'out' and len(filepath.split('.')) == 2]
        _filename = "/" + _filepath[0]

        # if _name not in done_list and os.path.getsize(root + _name + '/' + _filename) / (1024 * 1024) <= 10.0:
        ori_graph_path.append(_filename)
        name_list.append(_name)

    print("To process: ", len(name_list), "graphs")

    idx = 1
    for _name, _filename in zip(name_list, ori_graph_path):
        print('*'*60)
        print(idx, '/', len(name_list))
        print('Processing: '+_name+'... ' + time.ctime())

        try:
            print("Start loading...")
            G0 = gu.load_edgelist(root + _name + _filename, _name)
            G_after = gu.load_edgelist(root + _name + _filename, _name)
            print("Loading finished!")
            stats_G0 = G0.statistic_description()

            det_res = detect(G0, root+_name)
            num_fans = len(det_res[0])
            avg_fan_size = round(float(np.mean([len(det_res[0][k]) for k in det_res[0].keys()])), 2)
            num_conn2 = len(det_res[1])
            num_conn3 = len(det_res[2])
            num_traids = len(det_res[3])

            trans_res = transform(G_after, det_res, root+_name)

            stats_Gaf = G_after.statistic_description()
            v_redu = stats_G0[0] - stats_Gaf[0]
            e_redu = stats_G0[1] - stats_Gaf[1]

            time_G0 = time_Gaf = time_delta = time_redu = 0.0
            time_G0, _flag0 = sm.training(G0, root + _name + '/', o=0)
            time.sleep(7)
            if _flag0 == 0:
                raise Exception("Walking TOO LONG")
            time_G0 = round(time_G0, 3)
            time_Gaf, _flagaf = sm.training(G_after, root + _name + '/')
            time_Gaf = round(time_Gaf, 3)
            time.sleep(6)
            time_delta = time_G0 - time_Gaf
            time_redu = round(time_delta / time_G0*100, 2)

        except Exception as e:
            _info = time.ctime() + " " + _name + " " + repr(e)
            logging(_info)
        finally:
            out_report = [_name, stats_G0[0], stats_G0[1], time_G0, time_Gaf, time_delta, time_redu,
                          stats_Gaf[0], stats_Gaf[1], v_redu, e_redu,
                          num_fans, avg_fan_size, num_conn2, num_conn3, num_traids, det_res[-1], trans_res[-1]]
            with open('konect_SME_report-biggraph.csv', 'a', newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(out_report)

        idx += 1
        print('*' * 60)
        time.sleep(5)


if __name__ == '__main__':
    for i in range(0, 1):
        main()
