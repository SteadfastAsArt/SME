import pickle as pk


def serialization(obj, _path):
    with open(_path, 'wb') as f:
        pk.dump(obj, f)


def logging(_info):
    with open('_log.txt', 'a') as f:
        f.writelines(_info+'\n')
