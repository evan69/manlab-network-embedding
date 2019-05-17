import load_data
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

import csv

if __name__ == "__main__":
	trainX, trainY, testId, testX = load_data.build_dataset()
	
	# dt_model = DecisionTreeClassifier()
	dt_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0, class_weight='balanced')
	
	'''
	scores = cross_val_score(dt_model, trainX, trainY, cv = 10, scoring = 'accuracy')
	print (scores)
	print (sum(scores) / 10.0)
	'''
	dt_model.fit(trainX, trainY)
	testY = dt_model.predict(testX)
	print (testY)

	zipped = zip(testId, testY)

	with open("./result/random_forest_200_5_0_b.csv", 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['Node', 'Class'])
		for entry in zipped:
        		writer.writerow(entry)


