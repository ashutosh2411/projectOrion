import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import pandas
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix

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
	data = numpy.genfromtxt("ycc.csv" ,delimiter="," , autostrip = True )

	array = data
	print(array.shape)
	X = array[50:-50,in_columns]
	Y = array[50:-50,out_columns]
	print Y
	
	validation_size = 0.2
	seed =0
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
	for i in range(5):
		rf = RandomForestClassifier(n_estimators=80, criterion='gini', max_depth=None,
     					min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
    					 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
     					min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
     					random_state=None, verbose=0, warm_start=False, class_weight=None)
		rf.fit(X_train, Y_train)
		y_pred=rf.predict(X_validation)
		predictions = rf.score (X_validation,Y_validation)
		print 'RF : '+str(i) +' : '+str(predictions)
		#cnf_matrix = confusion_matrix(Y_validation, y_pred)
		#print cnf_matrix
	ecl = VotingClassifier(estimators=[('lr', lr),('a', knn),('r', lda),('l', nb), ('rf', svm),('lda',rf)], voting='hard')
	ecl.fit(X_train, Y_train)
	print accuracy_score(Y_validation, ecl.predict(X_validation))

in_columns  = range(2,7)+range(9,51)
out_columns = [62]

x_address = 'ycc.csv'

AllModels(x_address, in_columns, out_columns)