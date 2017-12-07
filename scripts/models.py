import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import pandas
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier
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
#	print(array.shape)
	X = array[50:-50,in_columns]
	Y = array[50:-50,out_columns]
#	print Y
	
	validation_size = 0.2
	seed =0
	#scoring = 'accuracy'

	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


	rf = []

	for i in range(1,5):
		rf.append(ExtraTreesClassifier(n_estimators=300, criterion='gini', max_depth=None, min_samples_split=2, 
						min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=i*12, 
						max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
						bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, 
						warm_start=False, class_weight=None))
		#cnf_matrix = confusion_matrix(Y_validation, y_pred)
		#print cnf_matrix
	l = []
	for i in range(len(rf)):
		l.append((str(i),rf[i]))
	lda = LinearDiscriminantAnalysis()
	l.append(('a',lda))
	l.append(('b',lda))
	ecl = VotingClassifier(estimators = l, voting = 'hard')
#	ecl = AdaBoostClassifier(base_estimator = rf[0])
	ecl.fit(X_train, Y_train)
	ret = accuracy_score(Y_validation, ecl.predict(X_validation))
	print ret
	return ret


in_columns  = range(2,51)
out_columns = [55]

x_address = 'ycc.csv'
s = 0.0
for i in range(25):
	s = s + AllModels(x_address, in_columns, out_columns)
print s/25





'''	lr = LogisticRegression()
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
	rf=RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=None,
		 					min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
							 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
		 					min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
		 					random_state=None, verbose=0, warm_start=False, class_weight=None)
	rf.fit(X_train, Y_train)
	print(rf.score(X_validation,Y_validation))
	et=ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=None, min_samples_split=2, 
						min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
						max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
						bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, 
						warm_start=False, class_weight=None)
	et.fit(X_train, Y_train)
	print(et.score(X_validation,Y_validation))
	#cnf_matrix = confusion_matrix(Y_validation, y_pred)
	#print cnf_matrix
'''	