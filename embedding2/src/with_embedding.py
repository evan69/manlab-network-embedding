import numpy as np
from scipy.sparse import lil_matrix
import random
import threading
import datetime
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression


local_test = True
train_path = '../data/facebook_edgelist'
test_path = '../data/facebook_test'
emb_path = '../data/facebook_struct2vec.emb'
predict_path = '../result/facebook_5.21.csv'
vertices = set()
test_edges = set()
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
            a, b = min(a, b), max(a, b)
            test_edges.add((a, b))
    return conj_matr


def read4local_test(pstv_num):
    """
    read for local test.
    这里将训练集的所有边edges（shuffle后）划分成两部分，①edges[:pstv_num]和②edges[pstv_num:]。
    ①用作本地测试集（的正例部分）；②用作本地训练集（的正例部分）

    :param pstv_num:
    :return: 返回的是 本地训练集的所有边的矩阵
    """
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
    test_edges = set(edges[:pstv_num])
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


def calc_auc(p_scores, n_scores):
    pstv_gth_ngtv = 0.0
    for p, p_score in enumerate(p_scores):
        for n, n_score in enumerate(n_scores):
            if p_score == n_score:
                pstv_gth_ngtv += 0.5
            elif p_score > n_score:
                pstv_gth_ngtv += 1
    # print(pstv_gth_ngtv)
    print('AUC is', pstv_gth_ngtv / len(p_scores) / len(n_scores))


def test_local(score_func):
    test_pstv_num = 500
    test_ngtv_num = 500
    conj = read4local_test(test_pstv_num)
    p_samples, n_samples = provide_sample(conj, test_edges, test_ngtv_num)

    p_scores = []
    n_scores = []
    print('calculating scores')
    for sp in p_samples:
        p_scores.append(score_func(conj, sp[0], sp[1]))
    print('calculating scores for negative samples')
    for sp in n_samples:
        n_scores.append(score_func(conj, sp[0], sp[1]))
    print('calculating AUC')
    calc_auc(p_scores, n_scores)


def predict(arg1, fn, func):
    test_score = []
    print('calculating scores fot test')
    for t in test_edges:
        test_score.append(func(arg1, t[0], t[1]))
    write(fn, test_edges, test_score)


def predict_all(fn, func, xs, origin_x):
    ys = func(xs)
    write(fn, origin_x, ys)


def thread_predict(conj, fn, func):  # fn = filename
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



if __name__ == '__main__':
    # 【读入edge_list和embedding表】
    if local_test is True:
        conj_mtrx = read4local_test(1000)  # 用于本地调试
        print('local test data read')
    else:
        conj_mtrx = read()  # 用于输出最后的预测结果
        print('online test data read')
    print('len(train_edges) =', len(train_edges))
    v_num, emb_dim, emb = load_emb(emb_path)
    print('embedding info: node_num =', v_num, ' embedding_dim =', emb_dim)

    # 【训练分类器模型】
    print('start training the model...')

    model = LinearRegression()

    sample_num = len(train_edges) * 4
    ps, ns = provide_sample(conj=conj_mtrx, pstv_set=train_edges, ngtv_num=sample_num // 4)
    xsl = [np.concatenate([emb[p[0]], emb[p[1]]]) for p in ps] + [np.concatenate([emb[p[1]], emb[p[0]]]) for p in ps] +\
          [np.concatenate([emb[n[0]], emb[n[1]]]) for n in ns] + [np.concatenate([emb[n[1]], emb[n[0]]]) for n in ns]
    Xs = np.array(xsl)
    print("Xs.shape:", Xs.shape)

    ys = np.concatenate([np.ones(sample_num // 2), np.zeros(sample_num // 2)])

    rand_ind = np.arange(sample_num)
    random.shuffle(rand_ind)
    Xs = Xs[rand_ind]
    ys = ys[rand_ind]

    model.fit(Xs, ys)
    print('model trained')

    score_func = model.predict

    # 【本地调试 或 预测结果输出】
    if local_test is True:  # 在本地测试分类器效果（AUC）

        test_num = 1000
        test_ps, test_ns = provide_sample(conj_mtrx, pstv_set=test_edges, ngtv_num=test_num // 2)
        test_Xs = np.array([np.concatenate([emb[p[0]], emb[p[1]]]) for p in test_ps] +
                           [np.concatenate([emb[n[0]], emb[n[1]]]) for n in test_ns])

        scores = score_func(test_Xs)

        calc_auc(scores[:test_num // 2], scores[test_num // 2:])

    else:  # 输出对比赛测试集的预测结果

        testlist = test_edges
        test_Xs = np.array([np.concatenate([emb[p[0]], emb[p[1]]]) for p in testlist])
        predict_all('', score_func, test_Xs, testlist)

