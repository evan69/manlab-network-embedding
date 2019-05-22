import load_data
import numpy as np
# import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import csv

if __name__ == "__main__":
	trainX, trainY, testId, testX = load_data.build_dataset("struc2vec.emb")
	trainX = np.array(trainX)
	trainY = np.array(trainY)
	# testX = np.array(testX)
	# testY = np.array(testY)
	
	# dt_model = DecisionTreeClassifier()
	knn = KNeighborsClassifier(n_neighbors = 100)
	dt_model = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=5, random_state=0, class_weight='balanced')
	lr_model= LogisticRegression(C = 0.3, penalty = 'l2')
	svm_model = SVC()
	mlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 8, 8, 8, 8, 4), random_state=1)
	gbdt_model = GradientBoostingClassifier()
	# xgbc = XGBClassifier()
	xgbc = XGBClassifier(#booster="gbliner",
	                      n_jobs=10,
	                      learning_rate=0.01,
	                      n_estimators=500,         # 树的个数--1000棵树建立xgboost
	                      max_depth=6,               # 树的深度
	                      min_child_weight = 1,      # 叶子节点最小权重
	                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
	                      subsample=0.75,             # 随机选择80%样本建立决策树
	                      colsample_btree=0.75,       # 随机选择80%特征建立决策树
	                      objective='multi:softmax', # 指定损失函数
	                      num_class=4,
	                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
	                      random_state=50,            # 随机数
	                      #alpha=10.0,
	                      #lambda=1.0
	                     )

	cur_model = xgbc

	scores = cross_val_score(cur_model, trainX, trainY, cv = 10, scoring = 'accuracy')
	print (scores)
	print (sum(scores) / 10.0)

	cur_model.fit(trainX, trainY)
	testY = cur_model.predict(testX)
	print (testY)

	zipped = zip(testId, testY)

	with open("./result/test.csv", 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['Node', 'Class'])
		for entry in zipped:
        		writer.writerow(entry)


