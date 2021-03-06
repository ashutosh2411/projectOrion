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
filename = 'master_nifty_manish.csv'
DO_PCA 					= 'n'
SPLIT_RANDOM			= 'n'				# 'y' for randomized split, anything else otherwise
DO_FEATURE_SELECTION 	= 'n'				# 'y' for yes, anything else otherwise

def r(l,u):
	return range(l, u+1)

#X_COLS 				= r(2,27)+r(28,43)+r(44,59)+r(60,75)+r(76,91)+r(92,107)+r(108,113)+r(114,119)			# both included; 2 means column C in excel
X_COLS = [37,38,39,41,45,46,47,48,49,58,74]

#################### don't confuse with name the split is in order-: train->
#################### test(for code simplicity)->validation(for test purpose)
#################### range of train, validation and test is by default validation + test size
X_TRAIN_START		= 2000
X_TRAIN_END			= 3000						# 50 means 50th row in excel
X_TEST_START		= 3000				# end is included
X_TEST_END			= 3200
Y_COLS 	 			= 121				# Y to be predicted											# 

CALCULATE_RETURNS	= 'y'				# 'y' for yes, anything else otherwise
RETURNS_COLS 		= 120		# To calculate returns. size same as Y_TO_PREDICT 	# If CALCULATE_RETURNS is not 'y', put any valid column in this one

N_FEATURES 			= None				# Number of features to select

#########################
Window_Size         = 100               #size of each rolling window
test_size           = 100				#size of each test set
#########################

COL_TOBE_SCALED 	= [32,6]			# 32 = OBV, 6 = Volume_today
##################### NO CHANGES BEYOND THIS POINT ##############################################
#################################################################################################
X_TRAIN_END 	= X_TRAIN_END + 1
X_TEST_END 		= X_TEST_END + 1

# Input: file name
def MAIN(file):
	data 		= np.genfromtxt(file ,delimiter = ',' , autostrip = True)
	for i in range (10):
		if SPLIT_RANDOM == 'y':
			X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data[X_TRAIN_START:X_TRAIN_END,X_COLS], data[X_TRAIN_START:X_TRAIN_END,[Y_COLS,RETURNS_COLS]], test_size=.2, random_state = 0)
		else:
			###################sequential split of train validation, test#################### 
			X_train 	 = data[X_TRAIN_START + i*Window_Size : X_TRAIN_END + i*Window_Size       , X_COLS]
			Y_train 	 = data[X_TRAIN_START + i*Window_Size : X_TRAIN_END + i*Window_Size       , [Y_COLS,RETURNS_COLS]]
			X_test 		 = data[X_TEST_START  + i*Window_Size : X_TEST_END  + i*Window_Size       , X_COLS]
			Y_test 		 = data[X_TEST_START  + i*Window_Size : X_TEST_END  + i*Window_Size       , [Y_COLS,RETURNS_COLS]]	
			X_validation = data[X_TEST_END    + i*Window_Size : X_TEST_END  + i*Window_Size + 100 , X_COLS]
			Y_validation = data[X_TEST_END    + i*Window_Size : X_TEST_END  + i*Window_Size + 100 , [Y_COLS,RETURNS_COLS]]

		X_train			= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(X_train)
		Y_train			= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(Y_train)
		X_test			= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(X_test)
		Y_test			= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(Y_test)
		X_validation	= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(X_validation)
		Y_validation	= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(Y_validation)
		
		scaler		 = MinMaxScaler().fit(X_train)
		X_train 	 = scaler.transform(X_train)
		X_test 		 = scaler.transform(X_test)
		X_validation = scaler.transform(X_validation)
		
		if DO_PCA == 'y':
			X       = PCA(n_components=4, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None).fit_transform(np.vstack((X_train,X_test)))
			X_train = X[:len(X_train)]
			X_test  = X[len(X_train):]
		
		Abs_train 	    = Y_train[:,1]
		Abs_test 	    = Y_test[:,1]
		Abs_validation 	= Y_validation[:,1]
		
		RunAllModels(Abs_train, Abs_test, Abs_validation, X_train, Y_train[:,0], X_test, Y_test[:,0], X_validation,Y_validation[:,0],i)

############## not relevant just for test#####################
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
def RunAllModels(Abs_train, Abs_test, Abs_validation ,X_train, Y_train, X_test, Y_test, X_validation,Y_validation, window_number):
	#RunLR (Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LR_')
	#RunLDA(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LDA')
	#RunLAS(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LAS')
	#RunRID(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'RID')
	#RunNB (Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'NB_')
	#RunKNN(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'KNN')
	#s = 0.0
	RunSVM(Abs_train, Abs_test, Abs_validation, X_train, Y_train, X_test, Y_test,X_validation,Y_validation, 'SVM', window_number)
		#q = RunRF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'RF_')
	#s = s + p
		#s1 = s1 + q
	#print('average svm ' + str(s/10))
	#print('average rf ' + str(s/10))
	#RunRF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'RF_')
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

def ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, 	actual_dist, need_print):
	per_train, acc_train 	= ComputeAccuracyForOne(cnf_mat_train)
	per_test, acc_test 		= ComputeAccuracyForOne(cnf_mat_test)
	if(need_print == 1):
		print (name + '_dist_actual_total          : ' + '%.3f %%,\t %.3f %%' %actual_dist)
		print (name + '_dist_pred_train            : ' + '%.3f %%,\t %.3f %%' %per_train)
		print (name + '_dist_pred_test             : ' + '%.3f %%,\t %.3f %%' %per_test)
		print (name + '_accuracy_train_[T,+,-]     : ' + '%.3f %%,\t %.3f %%,\t %.3f %%' %acc_train)
		print (name + '_accuracy_test__[T,+,-]     : ' + '%.3f %%,\t %.3f %%,\t %.3f %%' %acc_test)
	return (per_test, per_train, acc_test, acc_train)

def ComputeReturns(Abs_test, Abs_train, pred_test, pred_train,  Y_test, Y_train, name, need_print):
	ret_total_test, ret_cor_incor_test = Returns(Abs_test, pred_test, Y_test)
	ret_total_train, ret_cor_incor_train = Returns(Abs_train, pred_train, Y_train)

	if(need_print == 1):
		print ('	')
		print (name+'_ret.p.t_train_[T,+,-]      : '+'%.3f %%,\t %.3f %%,\t %.3f %%'%ret_total_train)
		print (name+'_ret.p.t_train_[cor, incor] : '+'%.3f %%,\t %.3f %%'%ret_cor_incor_train)
		print (name+'_ret.p.t_test_[T,+,-]       : '+'%.3f %%,\t %.3f %%,\t %.3f %%'%ret_total_test)
		print (name+'_ret.p.t_test_[cor, incor]  : '+'%.3f %%,\t %.3f %%'%ret_cor_incor_test)

	return (ret_total_train,ret_cor_incor_train, ret_total_test,ret_cor_incor_test)

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
def custom_threshold(prob_val,distribution):
	distribution = list(distribution)
	distribution[0] = distribution[0]/100.0
	distribution[1] = distribution[1]/100.0
	prob_val = np.asarray(prob_val)
	minus_prob = prob_val[:,0]
	plus_prob = prob_val[:,2]
	plus_prob = np.sort(plus_prob)
	minus_prob = np.sort(minus_prob)
	plus_thres = plus_prob[ int(np.ceil(len(plus_prob)*distribution[1] + (1-distribution[1])*0.75*len(plus_prob)))]
	minus_thres = minus_prob[int(np.ceil(len(minus_prob)*distribution[0] + (1-distribution[0])*0.75*len(plus_prob)))]
	print(minus_thres,plus_thres)
	return([minus_thres, plus_thres])
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
	model 				= LogisticRegressionCV(Cs=10, fit_intercept=True, cv=None, dual=False, 
							penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, 
							class_weight=None, n_jobs=1, verbose=0, refit=True, intercept_scaling=1.0, 
							multi_class='ovr', random_state=None)
	relevant_features 	= FeatureSelection(X_train, Y_train, model, N_FEATURES)
	X_train_ 			= X_train[:,relevant_features]
	X_test_				= X_test[:,relevant_features]
	model.fit(X_train_, Y_train)
#	print model.coef_
		
#	predicted_prob = model.predict_proba(X_test_)
#	class_label = [0]*len(predicted_prob) 
#	for i in range(len(predicted_prob[0])):
#		if(predicted_prob[0,i] < 0.505):
#			class_label[i] = -1
#		else:
#			class_label[i] = 1
#	pred_test = class_label
	pred_test	= model.predict(X_test_)
	pred_train 		= model.predict(X_train_)
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	if CALCULATE_RETURNS == 'y':
		returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
	print ('------------------------------------------')

def RunLDA(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	model 				= LinearDiscriminantAnalysis()
	relevant_features 	= FeatureSelection(X_train, Y_train, model, N_FEATURES)
	X_train_			= X_train[:,relevant_features]
	X_test_				= X_test[:,relevant_features]
	model.fit(X_train_, Y_train)
	pred_test 		= model.predict(X_test_)
	pred_train 	= model.predict(X_train_)
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	if CALCULATE_RETURNS == 'y':
		returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
	print ('------------------------------------------')

def RunLAS(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	model 				= Lasso()
	relevant_features 	= FeatureSelection(X_train, Y_train, model, N_FEATURES)
	X_train_			= X_train[:,relevant_features]
	X_test_				= X_test[:,relevant_features]
	model.fit(X_train_, Y_train)
	pred_test 		= np.sign(model.predict(X_test_))
	pred_train 		= np.sign(model.predict(X_train_))
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	if CALCULATE_RETURNS == 'y':
		returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
	print ('------------------------------------------')

def RunRID(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	model 				= Ridge()
	relevant_features 	= FeatureSelection(X_train, Y_train, model, N_FEATURES)
	X_train_			= X_train[:,relevant_features]
	X_test_				= X_test[:,relevant_features]
	model.fit(X_train_, Y_train)
	pred_test 		= np.sign(model.predict(X_test_))
	pred_train 		= np.sign(model.predict(X_train_))
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	if CALCULATE_RETURNS == 'y':
		returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
	print ('------------------------------------------')

def RunNB(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	model 			= GaussianNB()
	model.fit(X_train, Y_train)
	pred_test 	= model.predict(X_test)
	pred_train 	= model.predict(X_train)
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	if CALCULATE_RETURNS == 'y':
		returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
	print ('------------------------------------------')

def RunKNN(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	model 			= KNeighborsClassifier()
	model.fit(X_train, Y_train)
	pred_test 	= model.predict(X_test)
	pred_train 	= model.predict(X_train)
	
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	if CALCULATE_RETURNS == 'y':
		returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
	print ('------------------------------------------')
####################### IMPORTANT NOTE ####################################################
#######for the time being assume validation to be test and test to be validation###########
##########that has been done to keep rest of the code same#################################
##########       train is first 1000, test is next 200 and validation is next 100 from test_end
def RunSVM(Abs_train, Abs_test, Abs_validation, X_train, Y_train,  X_test, Y_test, X_validation, Y_validation, name, window_number):
	G_range_ = [0.001,0.005,0.01,0.05,0.1,0.15,0.25,0.28,0.75,1]+[10,20,40]
	C_range  = [0.8,1,2,6,7,8,10,50,100,1000,2500]
	G_range  = [1.0/i for i in G_range_]

	#C_range = [1000]
	c_array = []
	g_array = [] 

	#actual distribution in train and test set i.e. first 1200
	actual_dist_array = []
	
	#distribution in predicted test and train 
	predicted_test_array  = []
	predicted_train_array = []
	
	#accuracy in predicted test and train
	predicted_train_acc_array = []
	predicted_test_acc_array  = []
	
	#return per_trade for predicted train and test
	ret_pt_tot_train     = []
	ret_pt_cor_inc_train = []
	ret_pt_tot_test      = []
	ret_pt_cor_inc_test  = []

	#loop to iterate C and gamma for radial basis function
	for c in C_range:
		for g in G_range:

			model 			= SVC(C = c, kernel = 'rbf', gamma = g)
			model.fit(X_train, Y_train)
		
			pred_train 	= model.predict(X_train)
			pred_test 	= model.predict(X_test)
	
			cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
			cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
			actual_dist 	= ComputeDistribution(Y_train, [])		
			accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist,0)
			
			##############        change conditions accordingly          ####################
			
			#conditions
			# accuracy[0][0] is percentage plus in test(i.e. validation here)
			# accuracy[0][1] is percentage minus in test(i.e. validation here)
			# accuracy[1][0] is percentage plus in train
			# accuracy[2][0] is total accuracy test(i.e. validation here) ([2][1] and [2][2] consecutively plus)
			# accuracy[3][0] is percentage plus in train
			#putting condition over plus accuracy and minus accuracy
			if  (accuracy[2][0] > 52) and (accuracy[0][0] < 65) and (accuracy[3][0] < 75) :
				
				returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name,0)
				
				c_array.append(c)
				g_array.append(g)
				
				actual_dist_array.append(list(actual_dist))
				
				predicted_test_array.append(list(accuracy[0]))
				predicted_train_array.append(list(accuracy[1]))
				
				predicted_test_acc_array.append(list(accuracy[2])) 
				predicted_train_acc_array.append(list(accuracy[3]))
				
				ret_pt_tot_train.append(list(returns[0]))
				ret_pt_tot_test.append(list(returns[2]))
				ret_pt_cor_inc_train.append(list(returns[1]))
				ret_pt_cor_inc_test.append(list(returns[3]))
				
	
	print(' ')
	print ('------------------------------------------')
	print ('------------------------------------------')
	print ('------------------------------------------')
	
	#################################################################
	#transposing all the valus to fit in csv
	c_array = np.asarray(c_array).T
	g_array = np.asarray(g_array).T
	ret_pt_tot_train = np.asarray(ret_pt_tot_train).T
	ret_pt_tot_test = np.asarray(ret_pt_tot_test).T
	ret_pt_cor_inc_train = np.asarray(ret_pt_cor_inc_train).T
	ret_pt_cor_inc_test = np.asarray(ret_pt_cor_inc_test).T
	predicted_train_array = np.asarray(predicted_train_array).T
	predicted_train_acc_array = np.asarray(predicted_train_acc_array).T
	predicted_test_acc_array = np.asarray(predicted_test_acc_array).T
	predicted_test_array = np.asarray(predicted_test_array).T
	actual_dist_array = np.asarray(actual_dist_array).T
	######################################################################


	if(len(predicted_test_acc_array) > 0):
		#total accuracy of predicted test
		test_accuracy = predicted_test_acc_array[0,:]
	
		#index of sorted test_accuracy(200 after 1000)
		sorted_test_accuracy_arg = np.argsort(test_accuracy)
	
		index_for_best_c_g = sorted_test_accuracy_arg[len(sorted_test_accuracy_arg) - 1 ]
		c = c_array[index_for_best_c_g]
		g = g_array[index_for_best_c_g]

		print("best c " + str(c))
		print("best gamma " + str(g))

		model = SVC(C = c, gamma = g, kernel = 'rbf')

		X_train = np.vstack((X_train,X_test))
		Y_train = np.hstack((Y_train,Y_test))
		Abs_train = np.hstack((Abs_train,Abs_test))


		model.fit(X_train, Y_train)
		pred_train = model.predict(X_train)
		pred_validation = model.predict(X_validation)
		cnf_mat_test 	= GenerateCnfMatrix(pred_validation, Y_validation)
		cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
		####actual distribution over all the data passed train + test + validation
		actual_dist 	= ComputeDistribution(Y_train, Y_validation)	
	
		accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist,1)
		print('----------------------------')
		ComputeReturns(Abs_validation, Abs_train, pred_validation, pred_train, Y_validation, Y_train, name,1)
		print ('------------------------------------------')
		print ('------------------------------------------')
		print ('------------------------------------------')
		#out = out.T
		#header = ['c','gamma','dist_plus_actual','dist_minus_act','pred_plus_train','pred_minus_train','pred_plus_test','pred_minus_test','pred_tain_accuracy_tot','pred_train_acc_plus','pred_train_acc_minus','pre_test_acc_tot','pred_test_acc_plus','pred_test_acc_minus','ret_pt_tot_train','ret_pt_tot_plus','ret_pt_train_minus','ret_pt_cor_train','ret_pt_inc_train','rt_pt_tot_test','rt_pt_plus_test','rt_pt_minus_test','rt_pt_cor_test','ret_pt_inc_test']	
		#header = np.asarray(header)
		#out = np.vstack((header,out))
	else:
		print('----no best c and gamma----------------')

	#
	out = np.vstack((c_array,g_array,actual_dist_array,predicted_train_array,predicted_test_array,predicted_train_acc_array,predicted_test_acc_array,ret_pt_tot_train,ret_pt_cor_inc_train,ret_pt_tot_test,ret_pt_cor_inc_test))
	name_of_file = 'nifty_till_2017'+str(c)+'_' + str(g)+ '_' + str(window_number)+ '.csv'
	#np.savetxt(name_of_file, out.T, delimiter=",")

	print '----------------moving to other window-------------'
	
	return(accuracy[2][0])

#to calculate accuracy when test is predicted over a threshold
def calculate_acc(y_pred, 	y_true):
	tot_plus = 0.0
	cor_plus = 0.0
	tot_minus = 0.0
	cor_minus = 0.0
	for i in range(len(y_pred)):
		if(y_pred[i] >0 ):
			tot_plus = tot_plus+1
			if(y_true[i] > 0):
				cor_plus = cor_plus+1
		elif(y_pred[i]<0):
			tot_minus = tot_minus+1
			if(y_true[i]<0):
				cor_minus = cor_minus + 1
	print("accuracy when given a threshold")
	print("total +1 " + str(tot_plus))
	print("total -1 " + str(tot_minus))
	print("correct +1 ", str(cor_plus/tot_plus))
	print("correct -1 ", str(cor_minus/tot_minus))
	print("-------------------------------------")

def RunRF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	custom_score = make_scorer(accuracy_score)
	param_grid =  [{ 'min_samples_split': [3*i for i in range(2,6)], 'min_samples_leaf': [4*i for i in range(6,10)]}]
	model = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=40, min_samples_split=2, min_samples_leaf=1, 
								min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, 
								oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
	clf = model_selection.GridSearchCV(model, param_grid, scoring=None, cv = TimeSeriesSplit(n_splits = 5))
	clf.fit(X_train, Y_train)
	x = (clf.best_params_ )

	print x
	
	model.set_params(**x)
#	model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=20, min_impurity_decrease=0.002, min_impurity_split=None, class_weight=None, presort=False)
	relevant_features 	= FeatureSelection(X_train, Y_train, model, N_FEATURES)
#	model 				= RandomForestClassifier(n_estimators=300, criterion='entropy',random_state = 0)
	X_train_			= X_train[:,relevant_features]
	X_test_				= X_test[:,relevant_features]
	actual_dist 	= ComputeDistribution(Y_train, [])
	model.fit(X_train_, Y_train)
	pred_prob_train = model.predict_proba(X_train_)
	pred_prob_test = model.predict_proba(X_test_)
#	threshold = custom_threshold(pred_prob_train,actual_dist)
#	pred_test = [0]*len(Y_test)
#	for i in range(len(Y_test)):
#		if(pred_prob_test[i][0] > threshold[0]):
#			pred_test[i] = -1
#		elif(pred_prob_test[i][2] > threshold[1]):
#			pred_test[i] = 1
#	pred_thres_array = np.vstack((X_test_.T, Abs_test, Y_test, pred_test))
#	np.savetxt("pre_out_nifty_200.csv", pred_thres_array.T, delimiter=",")
#	calculate_acc(pred_test, Y_test)

	pred_test 		= model.predict(X_test_)
	pred_train 		= model.predict(X_train_)
	actual_dist 	= ComputeDistribution(Y_train, Y_test)
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	#print np.mean(cross_val_score(model, X_train, Y_train, cv=100))
	if CALCULATE_RETURNS == 'y':
		returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
	print ('------------------------------------------')
	return(accuracy[2][0])

def RunERF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	model 				= ExtraTreesClassifier()
	relevant_features 	= FeatureSelection(X_train, Y_train, model, N_FEATURES)
	X_train_			= X_train[:,relevant_features]
	X_test_				= X_test[:,relevant_features]
	model.fit(X_train_, Y_train)
	print np.argsort(model.feature_importances_)
	pred_test 		= model.predict(X_test_)
	pred_train 		= model.predict(X_train_)
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
	if CALCULATE_RETURNS == 'y':
		returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
	print ('------------------------------------------')
#	return s/len(X_train)

def FeatureSelection(X_train, Y_train, model, n):
	if DO_FEATURE_SELECTION == 'y':
		relevant_features = RFE(model, n_features_to_select=n, step=1000, verbose=0).fit(X_train, Y_train).support_
	else:
		relevant_features = [True]*X_train.shape[1]
	return relevant_features

# call to the main function
MAIN(filename)