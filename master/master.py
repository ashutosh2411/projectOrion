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

####################################
def r(l,u):
	return range(l,u+1)

#################################################################################################
################# CHANGES TO BE MADE ONLY IN FOLLOWING PART #####################################
NUM_REDUNDANT_ROWS 	= 35				# The number of redundant rows
NUM_REDUNDANT_COLS 	= 2					# The number of redundant columns
COL_TOBE_SCALED 	= [32,6]			# 32 = OBV, 6 = Volume_today
TRAIN_TEST_FACTOR 	= 0.2				# Test : Train ratio
CHUNK_SIZE 			= 10000000.0		# Train + Test chunks' size
X_COLS 				= r(50,50)			# Last index of Features
Y_COLS 	 			= [64]				# Y to be predicted
R_COLS 				= [62]				# To calculate returns. size same as Y_TO_PREDICT
N_FEATURES 			= None				# Number of features to select
##################### NO CHANGES BEYOND THIS POINT ##############################################
#################################################################################################

# Input: file name
def MAIN(file):
	data 		= np.genfromtxt(file ,delimiter = ',' , autostrip = True )[NUM_REDUNDANT_ROWS:]
	data 		= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(data)
#	data 		= scale(data[:,COL_TOBE_SCALED])
	n_splits 	= math.ceil(len(data)/CHUNK_SIZE)
	train, test = KSplit(data, n_splits, TRAIN_TEST_FACTOR)
	X_train 	= train[:,:,X_COLS]
	Y_train 	= train[:,:,Y_COLS]
	Abs_train 	= train[:,:,R_COLS]
	X_test 		= test[:,:,X_COLS]
	Y_test 		= test[:,:,Y_COLS]
	Abs_test 	= test[:,:,R_COLS]
	s = 0
	for ys in range(len(Y_COLS)):
		RunAllModels(Abs_train, Abs_test, X_train, Y_train[:,:,ys], X_test, Y_test[:,:,ys])
		print ''

def KSplit(data, n_splits, f):
	size 		= int(len(data)/n_splits)
	test_size  	= int(size*f)
	train_size 	= int(size - test_size)
	train 		= []
	test 		= []
	for i in range(int(n_splits)):
		train.append(data[i*size : i*size + train_size])
		test.append(data[i*size + train_size :(i + 1)*size])
	return(np.asarray(train), np.asarray(test))

def RunAllModels(Abs_train, Abs_test ,X_train, Y_train, X_test, Y_test):
	RunLR (Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LR_')
	RunLDA(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LDA')
	RunLAS(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LAS')
	RunRID(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'RID')
	RunNB (Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'NB_')
	RunKNN(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'KNN')
	RunSVM(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'SVM')
	RunERF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'RF_')
	RunERF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'ERF')

def GenerateCnfMatrix(Y_pred, Y_test):
	Y_p = []
	Y_t = []
	for i in range(len(Y_pred)):
		Y_p 	= np.hstack((Y_p,Y_pred[i]))
	for i in range(len(Y_test)):
		Y_t 	= np.hstack((Y_t,Y_test[i]))
	cnf_matrix 	= confusion_matrix(Y_t, Y_p, labels = [1,-1])
	return cnf_matrix

def ComputeAccuracy(cnf_mat_test, cnf_mat_train, name):
	per_train, acc_train 	= ComputeAccuracyForOne(cnf_mat_train)
	per_test, acc_test 		= ComputeAccuracyForOne(cnf_mat_test)
#	print name + '_dist_train: ' + str(per_train)
#	print name + '_dist_test : ' + str(per_test)
#	print name + '_acc_train : ' + str(acc_train)
	print name + '_acc_test  : ' + str(acc_test)
#	return (per_test, per_train, acc_test, acc_train)

def ComputeReturns(Abs_test, Abs_train, pred_test, pred_train,  Y_test, Y_train):
	Abs_test 	= OpenToList(Abs_test[:,:,0])
	Abs_train 	= OpenToList(Abs_train[:,:,0])
	pred_test 	= OpenToList(pred_test)
	pred_train 	= OpenToList(pred_train)
	Y_test 		= OpenToList(Y_test)
	Y_train 	= OpenToList(Y_train)
#	print 'returns_train : '+str (Returns(Abs_test, pred_test, Y_test))
#	print 'returns_test  : '+str (Returns(Abs_train, pred_train, Y_train))

def OpenToList(array):
	Y_p = []
	for i in range(len(array)):
		Y_p = np.hstack((Y_p,array[i]))
	return Y_p

def Returns(Abs, pred, Y):
	s0, s1, s2 = 0.0,0.0,0.0
	l = len(pred)
	for i in range(l):
		if pred[i] == Y[i]:
			s0 = s0 + pred[i]*Abs[i]
			if pred[i] == 1:
				s1 = s1 + Abs[i]
			else :
				s2 = s2 + Abs[i]
		else: 
			s0 = s0 - pred[i]*Abs[i]
	return s0/l, s1/l, s2/l

def ComputeAccuracyForOne(cnf_mat):	
#	[[tp, fn]
#	 [fp, tn]]
	tp, fn, fp, tn 		= GenerateTwoLabelCnfMatrix(cnf_mat).ravel()+(0.0,0.0,0.0,0.0)
	precision 			= tp/(tp + fp)
	recall 				= tp/(tp + fn)		# accuracy of plus
	specificity			= tn/(tn + fp)		# accuracy of minus
	accuracy_total 		= (tp + tn)/(tp + tn + fp + fn)	
	accuracy_plus 		= tp/(tp + fn)
	accuracy_minus 		= tn/(tn + fp)
	percent_plus		= (tp+fp)/(tp + tn + fp + fn)
	percent_minus		= (tn+fn)/(tp + tn + fp + fn)
	precent_list		= list((percent_plus, percent_minus))
	accuracy_list		= list((accuracy_total, accuracy_plus, accuracy_minus))
	return precent_list, accuracy_list

#################################################################################
def GenerateTwoLabelCnfMatrix(cnf_mat):
	return cnf_mat

def RunLR(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	s = 0
	pred_test 	= [0]*len(Y_test)
	pred_train 	= [0]*len(Y_train)
	for i in range(len(X_train)):
		model 				= LogisticRegression()
		relevant_features 	= FeatureSelection(X_train[i], Y_train[i], model, N_FEATURES)
		X_train_ 			= X_train[i,:,relevant_features].T
		X_test_				= X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		pred_test[i] 		= model.predict(X_test_)
		pred_train[i] 		= model.predict(X_train_)
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train)
	print '------------------------------------------'

def RunLDA(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	s = 0
	pred_test 	= [0]*len(Y_test)
	pred_train 	= [0]*len(Y_train)
	for i in range(len(X_train)):
		model 				= LinearDiscriminantAnalysis()
		relevant_features 	= FeatureSelection(X_train[i], Y_train[i], model, N_FEATURES)
		X_train_			= X_train[i,:,relevant_features].T
		X_test_				= X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		pred_test[i] 		= model.predict(X_test_)
		pred_train[i] 	= model.predict(X_train_)
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train)
	print '------------------------------------------'

def RunLAS(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	s = 0
	pred_test 	= [0]*len(Y_test)
	pred_train 	= [0]*len(Y_train)
	for i in range(len(X_train)):
		model 				= Lasso()
		relevant_features 	= FeatureSelection(X_train[i], Y_train[i], model, N_FEATURES)
		X_train_			= X_train[i,:,relevant_features].T
		X_test_				= X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		pred_test[i] 		= np.sign(model.predict(X_test_))
		pred_train[i] 		= np.sign(model.predict(X_train_))
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train)
	print '------------------------------------------'

def RunRID(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	s = 0
	pred_test 	= [0]*len(Y_test)
	pred_train 	= [0]*len(Y_train)
	for i in range(len(X_train)):
		model 				= Ridge()
		relevant_features 	= FeatureSelection(X_train[i], Y_train[i], model, N_FEATURES)
		X_train_			= X_train[i,:,relevant_features].T
		X_test_				= X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		pred_test[i] 		= np.sign(model.predict(X_test_))
		pred_train[i] 		= np.sign(model.predict(X_train_))
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train)
	print '------------------------------------------'

def RunNB(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	s = 0
	pred_test 	= [0]*len(Y_test)
	pred_train 	= [0]*len(Y_train)
	for i in range(len(X_train)):
		model 			= GaussianNB()
		model.fit(X_train[i], Y_train[i])
		pred_test[i] 	= model.predict(X_test[i])
		pred_train[i] 	= model.predict(X_train[i])
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train)
	print '------------------------------------------'

def RunKNN(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	s = 0
	pred_test 	= [0]*len(Y_test)
	pred_train 	= [0]*len(Y_train)
	for i in range(len(X_train)):
		model 			= KNeighborsClassifier()
		model.fit(X_train[i], Y_train[i])
		pred_test[i] 	= model.predict(X_test[i])
		pred_train[i] 	= model.predict(X_train[i])
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train)
	print '------------------------------------------'

def RunSVM(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	s = 0
	pred_test 	= [0]*len(Y_test)
	pred_train 	= [0]*len(Y_train)
	for i in range(len(X_train)):
		model 			= SVC()
		model.fit(X_train[i], Y_train[i])
		pred_test[i] 	= model.predict(X_test[i])
		pred_train[i] 	= model.predict(X_train[i])
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train)
	print '------------------------------------------'

def RunRF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	s = 0
	pred_test 	= [0]*len(Y_test)
	pred_train 	= [0]*len(Y_train)
	for i in range(len(X_train)):
		model 				= RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=None,
		 					min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
							 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
		 					min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
		 					random_state=None, verbose=0, warm_start=False, class_weight=None)
		relevant_features 	= FeatureSelection(X_train[i], Y_train[i], model, N_FEATURES)
		X_train_			= X_train[i,:,relevant_features].T
		X_test_				= X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		pred_test[i] 		= model.predict(X_test_)
		pred_train[i] 		= model.predict(X_train_)
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train)
	print '------------------------------------------'

def RunERF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	s = 0
	pred_test 	= [0]*len(Y_test)
	pred_train 	= [0]*len(Y_train)
	for i in range(len(X_train)):
		model 				= ExtraTreesClassifier()
		relevant_features 	= FeatureSelection(X_train[i], Y_train[i], model, N_FEATURES)
		X_train_			= X_train[i,:,relevant_features].T
		X_test_				= X_test[i,:,relevant_features].T
		model.fit(X_train_, Y_train[i])
		pred_test[i] 		= model.predict(X_test_)
		pred_train[i] 		= model.predict(X_train_)
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train)
	print '------------------------------------------'
	return s/len(X_train)

def FeatureSelection(X_train, Y_train, model, n):
	#relevant_features = RFE(model, n_features_to_select=n, step=1000, verbose=0).fit(X_train, Y_train).support_
	relevant_features = [True]*X_train.shape[1]
	return relevant_features

# call to the main function
MAIN('ycc.csv')