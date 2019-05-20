import numpy as np
from scipy.sparse import lil_matrix
import random
import threading
import datetime
from sklearn.svm import SVC

train_path = '../data/facebook_edgelist'
test_path = '../data/facebook_test'
emb_path = '../data/facebook.emb'
vertices = set()
test_edges = []
train_edges = set()
maxv = 0
emb = {}


def load_emb(filename):
    fp = open(filename, "r")
    lines = fp.readlines()
    fp.close()

    node_emb = dict()

    sp = lines[0].split()
    node_n = int(sp[0])
    demen = int(sp[1])

    lines = lines[1:]
    for line in lines:
        line = line[:-1]
        # arr = np.array(demen, dtype=np.float)
        sp = line.split(" ")
        vid = int(sp[0])
        arr = np.array([float(entry) for entry in sp[1:]], dtype=np.float)
        node_emb[vid] = arr

    return node_n, demen, node_emb


def read():
    global maxv, test_edges, train_edges
    with open(train_path) as f:
        for line in f:
            if line == '':
                continue
            ab = line.split(',')
            a = int(ab[0])
            b = int(ab[1])
            a, b = min(a, b), max(a, b)
            vertices.add(a)
            vertices.add(b)
            train_edges.add((a, b))
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


def read4local_test(pstv_num):
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
    test_edges = set(edges[: pstv_num])
    train_edges = set(edges[pstv_num:])
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


def provide_sample(conj, pstv_set, ngtv_num):
    psample = random.sample(pstv_set, ngtv_num)
    nsample = []
    while len(nsample) < ngtv_num:
        a = random.randint(1, maxv)
        b = random.randint(1, maxv)
        a, b = min(a, b), max(a, b)
        if conj[a, b] > 0 or (a, b) in test_edges:
            continue
        else:
            nsample.append((a, b))
    return psample, nsample


# def test_local():
#     # print(nbr_cnt(conj_mtrx, 1, 2))
#     conj = read4local_test(500)
#     p_samples, n_samples = provide_sample(conj, 500)
#     p_scores = []
#     n_scores = []
#     print('calculating scores')
#     for sp in p_samples:
#         p_scores.append(nbr_score(conj, sp[0], sp[1]))
#     print('calculating scores for negative samples')
#     for sp in n_samples:
#         n_scores.append(nbr_score(conj, sp[0], sp[1]))
#     print('calculating AUC')
#
def calc_auc(p_scores, n_scores):
    pstv_gth_ngtv = 0.0
    for p, p_score in enumerate(p_scores):
        for n, n_score in enumerate(n_scores):
            if p_score == n_score:
                pstv_gth_ngtv += 0.5
            elif p_score > n_score:
                pstv_gth_ngtv += 1
    print(pstv_gth_ngtv)
    print('AUC is', pstv_gth_ngtv / len(p_scores) / len(n_scores))


def predict(arg1, fn, func):
    test_score = []
    print('calculating scores fot test')
    for t in test_edges:
        test_score.append(func(arg1, t[0], t[1]))
    write(fn, test_edges, test_score)


def predict_all(fn, func, xs, origin_x):
    ys = func(xs)
    write(fn, origin_x, ys)


def thread_predict(conj, fn, func):
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
                test_score.append(func(conj, e[0], e[1]))
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


# on given edge list.
# count nbr: 0.93
# svm concatenate 2 vecs: 0.876688
# svm concatenate 2 vecs and symmtry: 0.8861, 0.883192, 0.893224(5k*2) 0.8735, 0.88092, 0.8852(2w*2)
# svm abs(a-b): 0.846652, 0.853272(1w samples), 0.85114(5k samples)
# svm 2k samples. linear 0.8347, poly(5) 0.83142 sigmoid 0.87

if __name__ == '__main__':
    # conj_mtrx = read4local_test(1000)
    conj_mtrx = read()
    v_num, dim, emb = load_emb(emb_path)
    print('data read')
    svc = SVC(C=2.5, kernel='sigmoid', degree=5, gamma='auto', coef0=0.5, shrinking=True,
              probability=False, tol=0.001, cache_size=200, class_weight=None,
              verbose=False, max_iter=-1, decision_function_shape='ovr',
              random_state=None)
    sample_num = 5000
    ps, ns = provide_sample(conj_mtrx, train_edges, sample_num // 4)
    # xsl = [np.concatenate([emb[p[0]], emb[p[1]]]) for p in ps] + [np.concatenate([emb[n[0]], emb[n[1]]]) for n in ns]
    xsl = [np.concatenate([emb[p[0]], emb[p[1]]]) for p in ps] + [np.concatenate([emb[p[1]], emb[p[0]]]) for p in ps] +\
          [np.concatenate([emb[n[0]], emb[n[1]]]) for n in ns] + [np.concatenate([emb[n[1]], emb[n[0]]]) for n in ns]
    # xsl = [np.abs(emb[p[0]] - emb[p[1]]) for p in ps] + [np.abs(emb[n[0]] - emb[n[1]]) for n in ns]
    Xs = np.array(xsl)
    ys = np.concatenate([np.ones(sample_num // 2), np.zeros(sample_num // 2)])
    svc.fit(Xs, ys)
    print('svm trained')
    # test_num = 1000
    # test_ps, test_ns = provide_sample(conj_mtrx, test_edges, test_num // 2)
    # test_Xs = np.array([np.concatenate([emb[p[0]], emb[p[1]]]) for p in test_ps] +
    #                    [np.concatenate([emb[n[0]], emb[n[1]]]) for n in test_ns])
    # # test_Xs = np.array([np.abs(emb[p[0]] - emb[p[1]]) for p in test_ps] +
    # #                    [np.abs(emb[n[0]] - emb[n[1]]) for n in test_ns])
    # scores = svc.decision_function(test_Xs)
    # calc_auc(scores[:test_num // 2], scores[test_num // 2:])

    testlist = test_edges
    test_Xs = np.array([np.concatenate([emb[p[0]], emb[p[1]]]) for p in testlist])
    predict_all('../result/facebook_svm_sig.csv', svc.decision_function, test_Xs, testlist)

    # predict(conj_mtrx, '../result/naive_test.res', nbr_score)

