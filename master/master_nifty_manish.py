import numpy as np
import math

# importing models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

#feature selection
from sklearn.feature_selection import RFE

# importing helper functions
from sklearn.preprocessing import Imputer, scale, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer,log_loss, f1_score, average_precision_score,explained_variance_score, log_loss
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.decomposition import PCA



####################################

#################################################################################################
################# CHANGES TO BE MADE ONLY IN FOLLOWING PART #####################################
filename = 'nifty_2000_2005_ycc.csv'
DO_PCA 					= 'n'
SPLIT_RANDOM			= 'n'				# 'y' for randomized split, anything else otherwise
DO_FEATURE_SELECTION 	= 'n'				# 'y' for yes, anything else otherwise

def r(l,u):
	return range(l, u+1)

#X_COLS 				= r(2,27)+r(28,43)+r(44,59)+r(60,75)+r(76,91)+r(92,107)+r(108,113)+r(114,119)			# both included; 2 means column C in excel
X_COLS = [37,38,36,48,47,45,57,40,73,46,44]
X_TRAIN_START		= 2050
X_TRAIN_END			= 3000						# 50 means 50th row in excel
X_TEST_START		= 3001					# end is included
X_TEST_END			= 3051
print X_TEST_END
Y_COLS 	 			= 121				# Y to be predicted											# 

CALCULATE_RETURNS	= 'y'				# 'y' for yes, anything else otherwise
RETURNS_COLS 		= 120		# To calculate returns. size same as Y_TO_PREDICT 	# If CALCULATE_RETURNS is not 'y', put any valid column in this one

N_FEATURES 			= None				# Number of features to select

COL_TOBE_SCALED 	= [32,6]			# 32 = OBV, 6 = Volume_today
##################### NO CHANGES BEYOND THIS POINT ##############################################
#################################################################################################
X_TRAIN_END 	= X_TRAIN_END + 1
X_TEST_END 		= X_TEST_END + 1

# Input: file name
def MAIN(file):
	data 		= np.genfromtxt(file ,delimiter = ',' , autostrip = True)
	if SPLIT_RANDOM == 'y':
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data[X_TRAIN_START:X_TRAIN_END,X_COLS], data[X_TRAIN_START:X_TRAIN_END,[Y_COLS,RETURNS_COLS]], test_size=.2, random_state = 0)
	else:
		X_train 	= data[X_TRAIN_START:X_TRAIN_END,X_COLS]
		Y_train 	= data[X_TRAIN_START:X_TRAIN_END,[Y_COLS,RETURNS_COLS]]
		X_test 		= data[X_TEST_START:X_TEST_END,X_COLS]
		Y_test 		= data[X_TEST_START:X_TEST_END,[Y_COLS,RETURNS_COLS]]	
	X_train		= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(X_train)
	Y_train		= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(Y_train)
	X_test		= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(X_test)
	Y_test		= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(Y_test)
	scaler		= MinMaxScaler().fit(X_train)
	X_train 	= scaler.transform(X_train)
	X_test 		= scaler.transform(X_test)
	if DO_PCA == 'y':
		X = PCA(n_components=4, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None).fit_transform(np.vstack((X_train,X_test)))
		X_train = X[:len(X_train)]
		X_test = X[len(X_train):]
	Abs_train 	= Y_train[:,1]
	Abs_test 	= Y_test[:,1]
	RunAllModels(Abs_train, Abs_test, X_train, Y_train[:,0], X_test, Y_test[:,0],data)
def my_own_accuracy(y_true, y_pred):
	plus = 1.0
	minus = 1.0
	for i in range(len(y_true)):
		if(y_pred[i]>0):
			plus = plus + 1.0
		else:
			minus = minus + 1.0
	#print(plus)
	#print(minus)
	if(minus/(minus+plus) < 0.3):
		print('---------minus is less----------')
		print(minus)
		print(plus)
		print(accuracy_score(y_true,y_pred))
		return(-1)
		print('-----------------------------')
	else:
		print('-----------enough minus------------------')
		print(minus)
		print(plus)
		print(accuracy_score(y_true,y_pred))
		return(accuracy_score(y_true,y_pred))
		print('-----------------------------')
def RunAllModels(Abs_train, Abs_test ,X_train, Y_train, X_test, Y_test,data):
	#RunLR (Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LR_')
	#RunLDA(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LDA')
	#RunLAS(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LAS')
	#RunRID(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'RID')
	#RunNB (Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'NB_')
	#RunKNN(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'KNN')
	'''s = 0.0
	for i in range(1):
		X_train 	= data[X_TRAIN_START + i*70:X_TRAIN_END+i*70,X_COLS]
		Y_train 	= data[X_TRAIN_START+i*70:X_TRAIN_END+i*70,[Y_COLS,RETURNS_COLS]]
		Abs_train 	= Y_train[:,1]
		Y_train = Y_train[:,0]
		X_test 		= data[X_TEST_START+i*70:X_TEST_END+i*70,X_COLS]
		Y_test 		= data[X_TEST_START+i*70:X_TEST_END+i*70,[Y_COLS,RETURNS_COLS]]	
		Abs_test 	= Y_test[:,1]
		Y_test = Y_test[:,0]
		p =  RunSVM(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'SVM')
		print p
		
		s = s + p
	print('average' + str(s/10))'''
	#RunSVM(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'SVM')
	RunRF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'RF_')
	#RunERF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'ERF')

def ComputeDistribution(Y_train, Y_test):
	lp = 0.0
	lm = 0.0
	Y_train = np.sign(Y_train)
	Y_test = np.sign(Y_test)
	plus = Y_train == [1]*len(Y_train)
	minus = Y_train == [-1]*len(Y_train)
	plusses = Y_train[plus]
	minuses = Y_train[minus]
	lp = lp + len(plusses) + 0.0
	lm = lm + len(minuses) + 0.0
	plus = Y_test == [1]*len(Y_test)
	minus = Y_test == [-1]*len(Y_test)
	plusses = Y_test[plus]
	minuses = Y_test[minus]
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
	print (name + '_dist_actual_total          : ' + '%.3f %%,\t %.3f %%' %actual_dist)
	print (name + '_dist_pred_train            : ' + '%.3f %%,\t %.3f %%' %per_train)
	print (name + '_dist_pred_test             : ' + '%.3f %%,\t %.3f %%' %per_test)
	print (name + '_accuracy_train_[T,+,-]     : ' + '%.3f %%,\t %.3f %%,\t %.3f %%' %acc_train)
	print (name + '_accuracy_test__[T,+,-]     : ' + '%.3f %%,\t %.3f %%,\t %.3f %%' %acc_test)
	return (per_test, per_train, acc_test, acc_train)

def ComputeReturns(Abs_test, Abs_train, pred_test, pred_train,  Y_test, Y_train, name):
	ret_total_test, ret_cor_incor_test = Returns(Abs_test, pred_test, Y_test)
	ret_total_train, ret_cor_incor_train = Returns(Abs_train, pred_train, Y_train)
	print ('	')
	print (name+'_ret.p.t_train_[T,+,-]      : '+'%.3f %%,\t %.3f %%,\t %.3f %%'%ret_total_train)
	print (name+'_ret.p.t_train_[cor, incor] : '+'%.3f %%,\t %.3f %%'%ret_cor_incor_train)
	print (name+'_ret.p.t_test_[T,+,-]       : '+'%.3f %%,\t %.3f %%,\t %.3f %%'%ret_total_test)
	print (name+'_ret.p.t_test_[cor, incor]  : '+'%.3f %%,\t %.3f %%'%ret_cor_incor_test)

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
l4