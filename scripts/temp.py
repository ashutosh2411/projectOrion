import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import pandas
from sklearn.ensemble import RandomForestClassifier


from os import listdir
from os.path import isfile, join
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def AllModels (file, in_columns, out_columns):		
	dataset = pandas.read_csv(file )
	datasetout = pandas.read_csv(file, usecols = out_columns)
	#dataset_ = dataset.sample(frac=1)
	dataset_ = dataset
	array = dataset_.values
	print(array.shape)
	X = array[0:5488,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29]]
	Y = array[0:5488,[31]]
	#print X[range(10),:]
	#print Y[range(10),:]
	validation_size = 0.2
	seed = 0
	#scoring = 'accuracy'

	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

	lr = LogisticRegression()
	lr.fit(X_train, Y_train)
	predictions = lr.predict (X_validation)
	print 'LR : ' + str(accuracy_score(Y_validation, predictions))

	lda = LinearDiscriminantAnalysis()
	lda.fit(X_train, Y_train)
	predictions = lda.predict (X_validation)
	print 'LDA: ' +str(accuracy_score(Y_validation, predictions))

	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	predictions = knn.predict (X_validation)
	print 'KNN: '+str(accuracy_score(Y_validation, predictions))
	
	classifier = RandomForestClassifier(n_estimators=60, criterion='gini', max_depth=None, 
						 min_weight_fraction_leaf=0.0, 
    					 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
     					min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
    					random_state=None, verbose=0, warm_start=False, class_weight=None)
	classifier.fit(X_train, Y_train)
	predictions = classifier.score (X_validation,Y_validation)
	print 'RF : ' +str(predictions)

	rf = DecisionTreeClassifier()
	rf.fit(X_train, Y_train)
	predictions = rf.predict (X_validation)
	print 'DF : ' +str(accuracy_score(Y_validation, predictions))

	nb = GaussianNB()
	nb.fit(X_train, Y_train)
	predictions = nb.predict (X_validation)
	print 'NB : '+str(accuracy_score(Y_validation, predictions))	

	svm = SVC()
	svm.fit(X_train, Y_train)
	predictions = svm.predict (X_validation)
	print 'SVM: '+str(accuracy_score(Y_validation, predictions))	
	print '--------------------'

in_columns  = range(30)
out_columns = [31]

x_address = '../data.csv'
AllModels(x_address, in_columns, out_columns)