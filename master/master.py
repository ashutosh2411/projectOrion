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

#################################################################################################
################# CHANGES TO BE MADE ONLY IN FOLLOWING PART #####################################
NUM_REDUNDANT_COLS 	= 2					# The number of redundant columns
def r(l,u):
	return range(l- NUM_REDUNDANT_COLS ,u+1- NUM_REDUNDANT_COLS)
X_COLS 				= r(2,27)+r(28,43)+r(44,59)+r(60,75)+r(76,91)+r(92,107)+r(108,113)+r(114,119)			# both included; 2 means column C in excel
X_TRAIN_START		= 1000
X_TRAIN_END			= 2400						# 50 means 50th row in excel
X_TEST_START		= 2401						# end is included
X_TEST_END			= 2900
Y_COLS 	 			= r(133,133)				# Y to be predicted											# 
RETURNS_COLS 		= r(132,132)				# To calculate returns. size same as Y_TO_PREDICT
FEATURE_SELECTION 	= 'n'				# 'y' for yes, anything else otherwise
COL_TOBE_SCALED 	= [32,6]			# 32 = OBV, 6 = Volume_today
TRAIN_TEST_FACTOR 	= 0.2				# Test : Train ratio
CHUNK_SIZE 			= 500000.0		# Train + Test chunks' size
N_FEATURES 			= None				# Number of features to select
##################### NO CHANGES BEYOND THIS POINT ##############################################
#################################################################################################
X_TRAIN_START=X_TRAIN_START-3 
X_TRAIN_END=X_TRAIN_END-2
X_TEST_START=X_TEST_START-3 
X_TEST_END=X_TEST_END-2
# Input: file name
def MAIN(file):
	data 		= np.genfromtxt(file ,delimiter = ',' , autostrip = True)
	data 		= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(data[2:])
#	data 		= scale(data[:,COL_TOBE_SCALED])
#	n_splits 	= math.ceil(len(data)/CHUNK_SIZE)
#	train, test = KSplit(data, n_splits, TRAIN_TEST_FACTOR)
	data = np.asarray([data])
	X_train 	= data[:,X_TRAIN_START:X_TRAIN_END,X_COLS]
	Y_train 	= data[:,X_TRAIN_START:X_TRAIN_END,Y_COLS]
	Abs_train 	= data[:,X_TRAIN_START:X_TRAIN_END,RETURNS_COLS]
	X_test 		= data[:,X_TEST_START:X_TEST_END,X_COLS]
	Y_test 		= data[:,X_TEST_START:X_TEST_END,Y_COLS]
	Abs_test 	= data[:,X_TEST_START:X_TEST_END,RETURNS_COLS]
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

def ComputeDistribution(Y_train, Y_test):
	lp = 0.0
	lm = 0.0
	for i in range(len(Y_train)):
		Y_train[i] = np.sign(Y_train[i])
		Y_test[i] = np.sign(Y_test[i])
		plus = Y_train[i] == [1]*len(Y_train[i])
		minus = Y_train[i] == [-1]*len(Y_train[i])
		plusses = Y_train[i,plus]
		minuses = Y_train[i,minus]
		lp = lp + len(plusses) + 0.0
		lm = lm + len(minuses) + 0.0
		plus = Y_test[i] == [1]*len(Y_test[i])
		minus = Y_test[i] == [-1]*len(Y_test[i])
		plusses = Y_test[i,plus]
		minuses = Y_test[i,minus]
		lp = lp + len(plusses)
		lm = lm + len(minuses)
	return (lp/(lp + lm))*100, (lm/(lp+lm))*100

def GenerateCnfMatrix(Y_pred, Y_test):
	Y_p = []
	Y_t = []
	for i in range(len(Y_pred)):
		Y_p 	= np.hstack((Y_p,Y_pred[i]))
	for i in range(len(Y_test)):
		Y_t 	= np.hstack((Y_t,Y_test[i]))
	cnf_matrix 	= confusion_matrix(Y_t, Y_p, labels = [1,-1])
	return cnf_matrix

def ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, 	actual_dist):
	per_train, acc_train 	= ComputeAccuracyForOne(cnf_mat_train)
	per_test, acc_test 		= ComputeAccuracyForOne(cnf_mat_test)
	print name + '_dist_actual_total          : ' + '%.3f %%,\t %.3f %%' %actual_dist
	print name + '_dist_pred_train            : ' + '%.3f %%,\t %.3f %%' %per_train
	print name + '_dist_pred_test             : ' + '%.3f %%,\t %.3f %%' %per_test
	print name + '_accuracy_train_[T,+,-]     : ' + '%.3f %%,\t %.3f %%,\t %.3f %%' %acc_train
	print name + '_accuracy_test__[T,+,-]     : ' + '%.3f %%,\t %.3f %%,\t %.3f %%' %acc_test
	return (per_test, per_train, acc_test, acc_train)

def ComputeReturns(Abs_test, Abs_train, pred_test, pred_train,  Y_test, Y_train, name):
	Abs_test 	= OpenToList(Abs_test[:,:,0])
	Abs_train 	= OpenToList(Abs_train[:,:,0])
	pred_test 	= OpenToList(pred_test)
	pred_train 	= OpenToList(pred_train)
	Y_test 		= OpenToList(Y_test)
	Y_train 	= OpenToList(Y_train)
	ret_total_test, ret_cor_incor_test = Returns(Abs_test, pred_test, Y_test)
	ret_total_train, ret_cor_incor_train = Returns(Abs_train, pred_train, Y_train)
	print '	'
	print name+'_ret.p.t_train_[T,+,-]      : '+'%.3f %%,\t %.3f %%,\t %.3f %%'%ret_total_train
	print name+'_ret.p.t_train_[cor, incor] : '+'%.3f %%,\t %.3f %%'%ret_cor_incor_train
	print name+'_ret.p.t_test_[T,+,-]       : '+'%.3f %%,\t %.3f %%,\t %.3f %%'%ret_total_test
	print name+'_ret.p.t_test_[cor, incor]  : '+'%.3f %%,\t %.3f %%'%ret_cor_incor_test

def OpenToList(array):
	Y_p = []
	for i in range(len(array)):
		Y_p = np.hstack((Y_p,array[i]))
	return Y_p

def Returns(Abs, pred, Y):
# s0: total per trade returns
# s1: +1 per trade returns
# s2: -1 per trade returns
# s3: total per trade returns when prediction is correct
# s4: total per trade returns when prediction is incorrect
	s0, s1, s2, s3, s4 = 0.0,0.0,0.0,0.0,0.0
	pred = np.sign(pred)
	l0, l1, l2, l3, l4 = 0.0,0.0,0.0,0.0,0.0
	for i in range(len(pred)):
		s0 = s0 + pred[i]*Abs[i]
		l0 = l0 + 1
		if pred[i] == 1:
			s1 = s1 + pred[i]*Abs[i]
			l1 = l1 + 1
		elif pred[i] == -1:
			s2 = s2 + pred[i]*Abs[i]
			l2 = l2 + 1
		if pred[i] == Y[i]:
			s3 = s3 + pred[i]*Abs[i]
			l3 = l3 + 1
		else:
			s4 = s4 + pred[i]*Abs[i]
			l4 = l4 + 1
	if l0 == 0:
		ret0 = 0
	else:
		ret0 = (s0/l0)*100
	if l1 == 0:
		ret1 = 0
	else:
		ret1 = (s1/l1)*100
	if l2 == 0:
		ret2 = 0
	else:
		ret2 = (s2/l2)*100
	if l3 == 0:
		ret3 = 0
	else:
		ret3 = (s3/l3)*100
	if l4 == 0:
		ret4 = 0
	else:
		ret4 = (s4/l4)*100
	return (ret0,ret1,ret2), (ret3,ret4)

def ComputeAccuracyForOne(cnf_mat):	
#	[[tp, fn]
#	 [fp, tn]]
	tp, fn, fp, tn 		= GenerateTwoLabelCnfMatrix(cnf_mat).ravel()+(0.0,0.0,0.0,0.0)
	precision 			= (tp/(tp + fp))*100
	recall 				= (tp/(tp + fn))*100
	specificity			= (tn/(tn + fp))*100
	accuracy_total 		= ((tp + tn)/(tp + tn + fp + fn))*100
	accuracy_plus 		= (tp/(tp + fp))*100
	accuracy_minus 		= (tn/(tn + fn))*100
	percent_plus		= ((tp+fp)/(tp + tn + fp + fn))*100
	percent_minus		= ((tn+fn)/(tp + tn + fp + fn))*100
	precent_list		= list((percent_plus, percent_minus))
	accuracy_list		= list((accuracy_total, accuracy_plus, accuracy_minus))
	return tuple(precent_list), tuple(accuracy_list)

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
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
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
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
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
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
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
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
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
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
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
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
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
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
	print '------------------------------------------'

def RunRF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	s = 0
	pred_test 	= [0]*len(Y_test)
	pred_train 	= [0]*len(Y_train)
	for i in range(len(X_train)):
		model 				= RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=5,
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
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
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
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
	print '------------------------------------------'
	return s/len(X_train)

def FeatureSelection(X_train, Y_train, model, n):
	if FEATURE_SELECTION == 'y':
		relevant_features = RFE(model, n_features_to_select=n, step=1000, verbose=0).fit(X_train, Y_train).support_
	else:
		relevant_features = [True]*X_train.shape[1]
	return relevant_features

# call to the main function
MAIN('../scripts/ICICIBANK_ycc.csv')