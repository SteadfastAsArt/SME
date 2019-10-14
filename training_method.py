from gensim.models import Word2Vec
import time
import random
import graph_utils as gu
from multiprocessing import cpu_count


def deepwalk(G, _filepath, o=1, num_walks_node=10, walk_length=80,
             representation_size=128, window_size=5,):
    """not going to deal with memory exceeding case"""
    output = _filepath + G.name

    print("Walking...")
    time_start = time.time()
    walks = gu.build_deepwalk_corpus(G, num_paths=num_walks_node, path_length=walk_length,
                                     alpha=0, rand=random.Random(0))  # alpha = 0: do not go back
    time_end = time.time()
    print('Walking time cost:', time_end - time_start)

    print("Training...")
    time_start = time.time()
    # with negative sampling: 5(default)
    model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, workers=cpu_count())
    time_end = time.time()

    print('Training vectors time cost:', time_end - time_start)

    if o == 1:
        model.wv.save_word2vec_format(output + '.dw.emb')
    else:
        model.wv.save_word2vec_format(output + '0.dw.emb')

    return time_end - time_start
