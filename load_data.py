# coding:utf8
# import sklearn
import numpy as np

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
		id = int(sp[0])
		arr = np.array([float(entry) for entry in sp[1:]], dtype=np.float)
		node_emb[id] = arr

	return node_n, demen, node_emb

def load_label(filename):
	fp = open(filename, "r")
	lines = fp.readlines()
	fp.close()

	node_label = dict()

	for line in lines:
		line = line[:-1]
		sp = line.split(",")
		id = int(sp[0])
		label = int(sp[1])
		node_label[id] = label

	return node_label

def build_dataset():
	dir = "./manlab-network-embedding1/"
	train_file = "airport_train"
	test_file = "airport_test"
	emb_file = "struc2vec.emb"

	node_n, demen, node_emb = load_emb(dir + emb_file)
	train_label = load_label(dir + train_file)
	# test_label = load_label(dir + test_file)

	trainX = [node_emb[id] for id in train_label.keys()]
	trainY = [train_label[id] for id in train_label.keys()]

	fp = open(dir + test_file, "r")
	lines = fp.readlines()
	fp.close()

	testId = [int(line[:-1]) for line in lines]
	testX = [node_emb[int(line[:-1])] for line in lines]

	return trainX, trainY, testId, testX

if __name__ == '__main__':
	trainX, trainY, testX = build_dataset()
	print (trainX, trainY, testX)
	print (trainX[99], trainY[99])

