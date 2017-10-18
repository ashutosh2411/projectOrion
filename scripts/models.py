import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import pandas

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
	dataset = pandas.read_csv(file , usecols = [0]+in_columns+out_columns)

	dataset_ = dataset.sample(frac=1)
	array = dataset_.values
	X = array[:,in_columns]
	Y = array[:,out_columns]
	#print X[range(10),:]
	#print Y[range(10),:]
	validation_size = 0.2
	seed = 7
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
	print 'RF : ' +str(accuracy_score(Y_validation, predictions))

	nb = GaussianNB()
	nb.fit(X_train, Y_train)
	predictions = nb.predict (X_validation)
	print 'NB : '+str(accuracy_score(Y_validation, predictions))	

	svm = SVC()
	svm.fit(X_train, Y_train)
	predictions = svm.predict (X_validation)
	print 'SVM: '+str(accuracy_score(Y_validation, predictions))	
	print '--------------------'

in_columns  = range(16)
out_columns = [16]

x_csv = [f for f in listdir('../datasets_pro') if isfile(join('../datasets_pro', f)) and f.endswith('.csv') and f.startswith('indi_')]
for x in x_csv:
	x_address = '../datasets_pro/' + x
	print x[5:-4]
	AllModels(x_address, in_columns, out_columns)