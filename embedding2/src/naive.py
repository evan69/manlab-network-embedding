import numpy as np
from scipy.sparse import lil_matrix
import random
import threading
import datetime


train_path = '../data/facebook_edgelist'
test_path = '../data/facebook_test'
vertices = set()
test_edges = []
train_edges = []
maxv = 0


def read():
    global maxv, test_edges, train_edges
    with open(train_path) as f:
        for line in f:
            if line == '':
                continue
            ab = line.split(',')
            a = int(ab[0])
            b = int(ab[1])
            vertices.add(a)
            vertices.add(b)
            train_edges.append((a, b))
            if a > maxv:
                maxv = a
            if b > maxv:
                maxv = b
    conj_matr = lil_matrix((len(vertices)+1, len(vertices)+1), dtype=np.int)
    for a, b in train_edges:
        conj_matr[a, b] = 1
        conj_matr[b, a] = 1

    with open(test_path) as f:
        for line in f:
            if line == '':
                continue
            ab = line.split(',')
            a, b = int(ab[0]), int(ab[1])
            test_edges.append((a, b))
    return conj_matr


def read4local_test(pstv_num):  # train_edges里的一部分用作test_edges，剩余的当train_edges
    global maxv, test_edges, train_edges
    edges = []
    with open(train_path) as f:
        for line in f:
            if line == '':
                continue
            ab = line.split(',')
            a = int(ab[0])
            b = int(ab[1])
            vertices.add(a)
            vertices.add(b)
            edges.append((a, b))
            if a > maxv:
                maxv = a
            if b > maxv:
                maxv = b
    conj_matr = lil_matrix((len(vertices)+1, len(vertices)+1), dtype=np.int)
    random.shuffle(edges)
    test_edges = edges[: pstv_num]
    train_edges = edges[pstv_num:]
    for a, b in train_edges:
        conj_matr[a, b] = 1
        conj_matr[b, a] = 1
    return conj_matr


def write(fn, edges, scores):
    with open(fn, 'w') as f:
        f.write('NodePair,Score\n')
        for edge, sc in zip(edges, scores):
            f.write(str(edge[0]) + '-' + str(edge[1]) + ',' + str(sc) + '\n')


def nbr_cnt(conj, a, b):
    a_nbr = set(conj[a].nonzero()[1])
    b_nbr = set(conj[b].nonzero()[1])
    # print(len(a_nbr), len(b_nbr))
    return a_nbr & b_nbr


def nbr_score(conj, a, b):
    return len(nbr_cnt(conj, a, b))


def provide_sample(conj, ngtv_num):
    psample = test_edges
    nsample = []
    for i in range(ngtv_num):
        a = random.randint(1, maxv)
        b = random.randint(1, maxv)
        a, b = min(a, b), max(a, b)
        if conj[a, b] > 0:
            i -= 1   # try once more
        else:
            nsample.append((a, b))
    return psample, nsample


def test_local():
    # print(nbr_cnt(conj_mtrx, 1, 2))
    conj = read4local_test(500)
    p_samples, n_samples = provide_sample(conj, 500)
    p_scores = []
    n_scores = []
    print('calculating scores')
    for sp in p_samples:
        p_scores.append(nbr_score(conj, sp[0], sp[1]))
    print('calculating scores for negative samples')
    for sp in n_samples:
        n_scores.append(nbr_score(conj, sp[0], sp[1]))
    print('calculating AUC')
    pstv_gth_ngtv = 0.0
    for p, p_score in enumerate(p_scores):
        for n, n_score in enumerate(n_scores):
            if p_score == n_score:
                pstv_gth_ngtv += 0.5
            elif p_score > n_score:
                pstv_gth_ngtv += 1
    print(pstv_gth_ngtv)
    print('AUC is', pstv_gth_ngtv / len(p_scores) / len(n_scores))


def predict(conj, fn):
    test_score = []
    print('calculating scores fot test')
    for t in test_edges:
        test_score.append(nbr_score(conj, t[0], t[1]))
    write(fn, test_edges, test_score)


def thread_predict(conj, fn):
    class PredictThread(threading.Thread):
        def __init__(self, name, tst_edgs, thread_fn):
            threading.Thread.__init__(self)
            self.tst_edges = tst_edgs
            self.fn = thread_fn
            self.name = name

        def run(self):
            test_score = []
            print('starting predict thread', self.name)
            for i, e in enumerate(self.tst_edges):
                test_score.append(nbr_score(conj, e[0], e[1]))
                if i % 100 == 0:
                    print(datetime.datetime.now().strftime('%d %H:%M:%S'), end='\t')
                    print(self.name, i, 'done out of', len(self.tst_edges))
            print('writing', self.name, 'to', self.fn)
            write(self.fn, self.tst_edges, test_score)
            print('predicting thread', self.name, 'done')
    thread_num = 32
    batch = len(test_edges) // thread_num
    threads = []
    for i in range(thread_num-1):
        t = PredictThread('predict' + str(i), test_edges[i*batch: (i+1)*batch], fn + str(i))
        threads.append(t)
        t.start()
    t = PredictThread('predict' + str(thread_num-1), test_edges[(thread_num-1)*batch:], fn + str(thread_num-1))
    t.start()
    threads.append(t)
    for t in threads:
        t.join()


if __name__ == '__main__':
    # test_local()
    conj_mtrx = read()
    print('data read')
    thread_predict(conj_mtrx, '../result/naive_test.res')
