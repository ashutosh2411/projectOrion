import numpy as np
import math

# importing models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

#feature selection
from sklearn.feature_selection import RFE

# importing helper functions
from sklearn.preprocessing import Imputer, scale
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

NUM_REDUNDANT_ROWS = 35
NUM_REDUNDANT_COLS = 2
COL_TOBE_SCALED = [32,6]			# 32 = OBV, 6 = Volume_today
TRAIN_TEST_FACTOR = 0.2
CHUNK_SIZE = 10000.0				# Train + Test chunks' size
MAX_X_COLS = 50
SRT_Y_COLS = MAX_X_COLS + 1
Y_TO_PREDICT = [53,56]
MAX_QUANT_LVL = 1

# modifying constants here: 
MAX_X_COLS = MAX_X_COLS - NUM_REDUNDANT_COLS + 1
Y_TO_PREDICT = np.subtract(Y_TO_PREDICT , [SRT_Y_COLS]*len(Y_TO_PREDICT))
SRT_Y_COLS = MAX_X_COLS

def MAIN(file):
	data = np.genfromtxt(file ,delimiter = "," , autostrip = True )[NUM_REDUNDANT_ROWS:]
	data = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0).fit_transform(data)
#	data = scale(data[:,COL_TOBE_SCALED])
	n_splits = math.ceil(len(data)/CHUNK_SIZE)
	train, test = KSplit(data, n_splits, TRAIN_TEST_FACTOR)
	X_train = train[:,:,0:MAX_X_COLS]
	Y_train = train[:,:,SRT_Y_COLS:]
	X_test = test[:,:,0:MAX_X_COLS]
	Y_test = test[:,:,SRT_Y_COLS:]
	s = 0
	for ys in Y_TO_PREDICT:
		RunAllModels(X_train, Y_train[:,:,ys], X_test, Y_test[:,:,ys])
		print '----------------------------'

def KSplit(data, n_splits, f):
	size = int(len(data)/n_splits)
	test_size  = int(size*f)
	train_size = int(size - test_size)
	train = []
	test = []
	for i in range(int(n_splits)):
		train.append(data[i*size : i*size + train_size])
		test.append(data[i*size + train_size : (i + 1)*size])
	return (np.asarray(train), np.asarray(test))

def RunAllModels(X_train, Y_train, X_test, Y_test):
	RunLR (X_train, Y_train, X_test, Y_test, 'LR_')
	RunLDA(X_train, Y_train, X_test, Y_test, 'LDA')
	RunLAS(X_train, Y_train, X_test, Y_test, 'LAS')
	RunRID(X_train, Y_train, X_test, Y_test, 'RID')
	RunNB (X_train, Y_train, X_test, Y_test, 'NB_')
	RunKNN(X_train, Y_train, X_test, Y_test, 'KNN')
	RunSVM(X_train, Y_train, X_test, Y_test, 'SVM')
	RunRF (X_train, Y_train, X_test, Y_test, 'RF_')
	RunERF(X_train, Y_train, X_test, Y_test, 'ERF')

def GenerateCnfMatrix(Y_pred, Y_test):
	Y_p = []
	Y_t = []
	for i in range(len(Y_pred)):
		Y_p = np.hstack((Y_p,Y_pred[i]))
	for i in range(len(Y_test)):
		Y_t = np.hstack((Y_t,Y_test[i]))
	cnf_matrix = confusion_matrix(Y_t, Y_p, labels = range(-MAX_QUANT_LVL,MAX_QUANT_LVL+1))
	return cnf_matrix

def ComputeAccuracy(cnf_mat, name):
	
	
def RunLR (X_train, Y_train, X_test, Y_test, name):
	s = 0
	predictions = [0]*len(Y_test)
	for i in range(len(X_train)):
		model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
			C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, 
			random_state=None, solver='liblinear', max_iter=100, 
			multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
		relevant_features = FeatureSelection(X_train[i], Y_train[i], model)
		X_train_=X_train[i,:,relevant_features].T
		X_test_=X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		predictions[i] = model.predict (X_test_)
		s = s + accuracy_score(Y_test[i], predictions[i])
	cnf_mat = GenerateCnfMatrix(predictions, Y_test)
	ComputeAccuracy(cnf_mat, name)
	print name + ' : ' + str(s/len(X_train))

def RunLDA (X_train, Y_train, X_test, Y_test, name):
	s = 0
	predictions = [0]*len(Y_test)
	for i in range(len(X_train)):
		model = LinearDiscriminantAnalysis()
		relevant_features = FeatureSelection(X_train[i], Y_train[i], model)
		X_train_=X_train[i,:,relevant_features].T
		X_test_=X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		predictions[i] = model.predict (X_test_)
		s = s + accuracy_score(Y_test[i], predictions[i])
	cnf_mat = GenerateCnfMatrix(predictions, Y_test)
	ComputeAccuracy(cnf_mat, name)
	print name + ' : ' + str(s/len(X_train))

def RunLAS (X_train, Y_train, X_test, Y_test, name):
	s = 0
	predictions = [0]*len(Y_test)
	for i in range(len(X_train)):
		model = Lasso()
		relevant_features = FeatureSelection(X_train[i], Y_train[i], model)
		X_train_=X_train[i,:,relevant_features].T
		X_test_=X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		predictions[i] = np.sign(model.predict (X_test_))
		s = s + accuracy_score(Y_test[i], predictions[i])
	cnf_mat = GenerateCnfMatrix(predictions, Y_test)
	ComputeAccuracy(cnf_mat, name)
	print name + ' : ' + str(s/len(X_train))

def RunRID (X_train, Y_train, X_test, Y_test, name):
	s = 0
	predictions = [0]*len(Y_test)
	for i in range(len(X_train)):
		model = Ridge()
		relevant_features = FeatureSelection(X_train[i], Y_train[i], model)
		X_train_=X_train[i,:,relevant_features].T
		X_test_=X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		predictions[i] = np.sign(model.predict (X_test_))
		s = s + accuracy_score(Y_test[i], predictions[i])
	cnf_mat = GenerateCnfMatrix(predictions, Y_test)
	ComputeAccuracy(cnf_mat, name)
	print name + ' : ' + str(s/len(X_train))

def RunNB (X_train, Y_train, X_test, Y_test, name):
	s = 0
	predictions = [0]*len(Y_test)
	for i in range(len(X_train)):
		model = GaussianNB()
		model.fit(X_train[i], Y_train[i])
		predictions[i] = model.predict (X_test[i])
		s = s + accuracy_score(Y_test[i], predictions[i])
	cnf_mat = GenerateCnfMatrix(predictions, Y_test)
	ComputeAccuracy(cnf_mat, name)
	print name + ' : ' + str(s/len(X_train))

def RunKNN (X_train, Y_train, X_test, Y_test, name):
	s = 0
	predictions = [0]*len(Y_test)
	for i in range(len(X_train)):
		model = KNeighborsClassifier()
		model.fit(X_train[i], Y_train[i])
		predictions[i] = model.predict (X_test[i])
		s = s + accuracy_score(Y_test[i], predictions[i])
	cnf_mat = GenerateCnfMatrix(predictions, Y_test)
	ComputeAccuracy(cnf_mat, name)
	print name + ' : ' + str(s/len(X_train))

def RunSVM (X_train, Y_train, X_test, Y_test, name):
	s = 0
	predictions = [0]*len(Y_test)
	for i in range(len(X_train)):
		model = SVC()
		model.fit(X_train[i], Y_train[i])
		predictions[i] = model.predict (X_test[i])
		s = s + accuracy_score(Y_test[i], predictions[i])
	cnf_mat = GenerateCnfMatrix(predictions, Y_test)
	ComputeAccuracy(cnf_mat, name)
	print name + ' : ' + str(s/len(X_train))

def RunRF (X_train, Y_train, X_test, Y_test, name):
	s = 0
	predictions = [0]*len(Y_test)
	for i in range(len(X_train)):
		model = RandomForestClassifier()
		relevant_features = FeatureSelection(X_train[i], Y_train[i], model)
		X_train_=X_train[i,:,relevant_features].T
		X_test_=X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		predictions[i] = model.predict (X_test_)
		s = s + accuracy_score(Y_test[i], predictions[i])
	cnf_mat = GenerateCnfMatrix(predictions, Y_test)
	ComputeAccuracy(cnf_mat, name)
	print name + ' : ' + str(s/len(X_train))
	return s/len(X_train)

def RunERF (X_train, Y_train, X_test, Y_test, name):
	s = 0
	predictions = [0]*len(Y_test)
	for i in range(len(X_train)):
		model = ExtraTreesClassifier()
		relevant_features = FeatureSelection(X_train[i], Y_train[i], model)
		X_train_=X_train[i,:,relevant_features].T
		X_test_=X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		predictions[i] = model.predict (X_test_)
		s = s + accuracy_score(Y_test[i], predictions[i])
	cnf_mat = GenerateCnfMatrix(predictions, Y_test)
	ComputeAccuracy(cnf_mat, name)
	print name + ' : ' + str(s/len(X_train))
	return s/len(X_train)

def FeatureSelection(X_train, Y_train, model):
	relevant_features = RFE(model, n_features_to_select=25, step=1000, verbose=0).fit(X_train, Y_train).support_
	return relevant_features

MAIN('../scripts/ICICIBANK_ycc.csv')