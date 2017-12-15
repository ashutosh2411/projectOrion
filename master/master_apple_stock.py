import numpy as np
import pandas

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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.decomposition import PCA

####################################

#################################################################################################
################# CHANGES TO BE MADE ONLY IN FOLLOWING PART #####################################
filename = 'AAPL_2001_to_2017_ycc.csv'
DO_PCA 					= 'n'
SPLIT_RANDOM			= 'n'				# 'y' for randomized split, anything else otherwise
DO_FEATURE_SELECTION 	= 'n'				# 'y' for yes, anything else otherwise
N_FEATURES 				= None				# Number of features to select
EXPONENTIAL_SMOOTHENING	= 'n'				# 'y' for yes, anything else otherwise
# Exponential smoothening for entire feature set. 
FEATURE_SELECTION_RF 	= 'n'
FILTER_RESULTS			= 'y'
FRACT_FEATURES			= .7
def r(l,u):
	return range(l, u+1)
#X_COLS 				= r(2,27)+r(28,43)+r(44,59)+r(60,75)+r(76,91)+r(92,107)+r(108,113)+r(114,119)			# both included; 2 means column C in excel
#X_COLS = [39,6,44,49,51,48,31,32,33,45,69,41,42,43,76,60,67,47,63,62,116,117,118,119]
X_COLS 				= r(31,56)
X_TRAIN_START		= 2267
X_TRAIN_END			= 2267+1000						# 50 means 50th row in excel
X_TEST_START		= 2267+1001							# end is included
X_TEST_END			= 2267+1200

Y_COLS 	 			= 66				# Y to be predicted													# 
print Y_COLS
CALCULATE_RETURNS	= 'y'				# 'y' for yes, anything else otherwise
RETURNS_COLS 		= 65				# To calculate returns. size same as Y_TO_PREDICT 	# If CALCULATE_RETURNS is not 'y', put any valid column in this one


COL_TOBE_SCALED 	= [32,6]			# 32 = OBV, 6 = Volume_today
##################### NO CHANGES BEYOND THIS POINT ##############################################
#################################################################################################
X_TRAIN_END 	= X_TRAIN_END + 1
X_TEST_END 		= X_TEST_END + 1
# Input: file name
def MAIN(file):
	data 		= np.genfromtxt(file ,delimiter = ',' , autostrip = True)
	s=0.0
	l=1
	windowSize = 10
	out = []
	for i in range(l):
		print i+1
		if SPLIT_RANDOM == 'y':
			print 'First shuffling, then splitting into test and train'
			X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data[X_TRAIN_START+10*i:X_TRAIN_END+10*i,X_COLS], data[X_TRAIN_START+10*i:X_TRAIN_END+10*i,[Y_COLS,RETURNS_COLS]], test_size=.05, random_state = 0)
		else:
			print 'Sequential splitting'
			X_train 	= data[X_TRAIN_START+windowSize*i:X_TRAIN_END+windowSize*i,X_COLS]
			Y_train 	= data[X_TRAIN_START+windowSize*i:X_TRAIN_END+windowSize*i,[Y_COLS,RETURNS_COLS]]
			X_test 		= data[X_TEST_START+windowSize*i:X_TEST_END+windowSize*i,X_COLS]
			Y_test 		= data[X_TEST_START+windowSize*i:X_TEST_END+windowSize*i,[Y_COLS,RETURNS_COLS]]	
		X_train		= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(X_train)
		Y_train		= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(Y_train)
		X_test		= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(X_test)
		Y_test		= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit_transform(Y_test)
		scaler		= MinMaxScaler().fit(X_train)
		X_train 	= scaler.transform(X_train)
		X_test 		= scaler.transform(X_test)
		if FEATURE_SELECTION_RF == 'y':
			print 'Doing feature selection using Feature importance as returned by RF classifier.'
			erf_features = SelectFeaturesRF(X_train, Y_train[:,0])
			print erf_features
			X_train 	= X_train[:,erf_features[:int(len(erf_features)*FRACT_FEATURES)]]
			X_test 		= X_test [:,erf_features[:int(len(erf_features)*FRACT_FEATURES)]]
#		X_train 	= X_train[:,[17 ,16 ,20 ,22 ,19 ,18 ,11  ,9 ,24 ,23  ,4  ,7  ,0 ,8,13,21 ,2,10,1,25,5]]
#		X_test 		= X_test [:,[17 ,16 ,20 ,22 ,19 ,18 ,11  ,9 ,24 ,23  ,4  ,7  ,0 ,8,13,21 ,2,10,1,25,5]]
		ewma = pandas.stats.moments.ewma
		if EXPONENTIAL_SMOOTHENING == 'y':
			X_train = ewma(X_train, com=.25)
			X_test 	= ewma(X_test, com=.25) 
		if DO_PCA == 'y':
			print 'doing feature selection'
			pca = PCA(n_components=10, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None).fit(X_train)
			X_train = pca.transform(X_train)
			X_test = pca.transform(X_test)
		Abs_train 	= Y_train[:,1]
		Abs_test 	= Y_test[:,1]
		a = RunAllModels(Abs_train, Abs_test, X_train, Y_train[:,0], X_test, Y_test[:,0])
	#	s = s+a[0]
	#	out.append(a[1])
	#	print 'average: '+ str(s/(i+1))
	
	#for i in range(l):
	#	out[i] = np.hstack(([0]*windowSize*(i+1), out[i] , [0]*windowSize*(l-i)))
	#out = np.asarray(out)
	#out_ = np.vstack((data[X_TEST_START- windowSize:X_TEST_END+l* windowSize, Y_COLS],out))
	#pandas.DataFrame(out_.T).to_csv('out.csv',index=False, header=None)

def RunAllModels(Abs_train, Abs_test ,X_train, Y_train, X_test, Y_test):
#	RunLR (Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LR_')
#	RunLDA(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LDA')
#	RunLAS(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'LAS')
#	RunRID(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'RID')
#	RunNB (Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'NB_')
#	RunKNN(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'KNN')
#	s = 0.0
#	l = 50
#	for i in range(1,l):
#		s = s + RunRF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'RF_')
#		print i
	RunSVM_(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'SVM')
#	return RunSVM(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'SVM')
#	print 'average : ' +str(s/l) 
#	return RunRF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'RF_')
#	RunERF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, 'ERF')



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
def RunSVM_(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	#for i in range(1,30):
	#custom_score = make_scorer(my_own_accuracy )
	G_range_ = [0.001,0.005,0.01,0.05,0.1,0.15,0.28,0.75,1]+range(10,140)
	#G_range_ = [1]
	G_range = [1.0/i for i in G_range_]
	C_range = [10,15,50,100,150,500,700,1000,2500,10000]
	#C_range = [1000]
	c_array = []
	g_array = [] 
	actual_dist_array = []
	predicted_test_array =[]
	predicted_train_array =[]
	predicted_train_acc_array =[]
	predicted_test_acc_array = []
	ret_pt_tot_train =[]
	ret_pt_cor_inc_train=[]
	ret_pt_tot_test=[]
	ret_pt_cor_inc_test=[]
	for c in C_range:
		for g in G_range:
			print 'c ' +str(c)
			print 'g ' +str(g)
	#param_grid =  [{ 'C': C_range,'kernel': ['rbf']}]
	#clf = model_selection.GridSearchCV(model, param_grid, cv = TimeSeriesSplit(n_splits = 5))
	#clf.fit(X_train, Y_train)
	#x = (clf.best_params_ )
	#print x
	#model.set_params(**x)
	
			model 			= SVC(C = c, kernel = 'rbf', gamma = g)

			model.fit(X_train, Y_train)
			pred_train 	= model.predict(X_train)
			pred_test 	= model.predict(X_test)
	
			cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
			cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
			actual_dist 	= ComputeDistribution(Y_train, Y_test)	
			
			accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)

			print(' ')
			print ('------------------------------------------')
			if accuracy[0][0] >= (actual_dist[0] -5) and accuracy[0][0] <= (actual_dist[0] +5) and accuracy[2][0] > 55 and accuracy[2][1] > 52 and accuracy[2][2] > 52:
				returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
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
	out = np.vstack((c_array,g_array,actual_dist_array,predicted_train_array,predicted_test_array,predicted_train_acc_array,predicted_test_acc_array,ret_pt_tot_train,ret_pt_cor_inc_train,ret_pt_tot_test,ret_pt_cor_inc_test))
	#out = out.T
	#header = ['c','gamma','dist_plus_actual','dist_minus_act','pred_plus_train','pred_minus_train','pred_plus_test','pred_minus_test','pred_tain_accuracy_tot','pred_train_acc_plus','pred_train_acc_minus','pre_test_acc_tot','pred_test_acc_plus','pred_test_acc_minus','ret_pt_tot_train','ret_pt_tot_plus','ret_pt_train_minus','ret_pt_cor_train','ret_pt_inc_train','rt_pt_tot_test','rt_pt_plus_test','rt_pt_minus_test','rt_pt_cor_test','ret_pt_inc_test']	
	#header = np.asarray(header)
	#out = np.vstack((header,out))
	np.savetxt("c_gaama_filtered.csv", out.T, delimiter=",")

def RunSVM(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	model 			= SVC(C=1000, kernel='rbf', gamma=1)
	gs = [1.0/i for i in range(1,100)]
	cs = [10**i for i in range(2,4)]
	param_grid =  [{'C': cs, 'gamma': gs}]
#	clf = model_selection.GridSearchCV(model, param_grid, scoring=None, fit_params=None, n_jobs=6, iid=True, 
#									refit='best_score_', cv = TimeSeriesSplit(n_splits = 5), verbose=0, pre_dispatch='2*n_jobs', 
#									error_score='raise', return_train_score='warn')
#	clf.fit(X_train, Y_train)
#	x = (clf.best_params_ )
#	print x
#	model.set_params(**x)
#	print model.get_params(deep=True)
	gs_ = [0.001,0.005,0.01,0.05,0.1,0.15,0.28,0.75,1]+range(10,140)
	gs = [1.0/i for i in gs_]
	cs = [10,15,50,100,150,500,700,1000,2500,10000]
	c_array = []
	g_array = [] 
	actual_dist_array = []
	predicted_test_array =[]
	predicted_train_array =[]
	predicted_train_acc_array =[]
	predicted_test_acc_array = []
	ret_pt_tot_train =[]
	ret_pt_cor_inc_train=[]
	ret_pt_tot_test=[]
	ret_pt_cor_inc_test=[]
	
	for c in cs:
		for g in gs:
			c_array.append(c)
			g_array.append(g)
			print 'c ' +str(c)
			print 'g ' +str(g)
	#param_grid =  [{ 'C': C_range,'kernel': ['rbf']}]
	#clf = model_selection.GridSearchCV(model, param_grid, cv = TimeSeriesSplit(n_splits = 5))
	#clf.fit(X_train, Y_train)
	#x = (clf.best_params_ )
	#print x
	#model.set_params(**x)
	
			model 			= SVC(C = c, kernel = 'rbf', gamma = g)

			model.fit(X_train, Y_train)
			pred_train 	= model.predict(X_train)
			pred_test 	= model.predict(X_test)
	
			cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
			cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
			actual_dist 	= ComputeDistribution(Y_train, Y_test)	
			actual_dist_array.append(list(actual_dist))
			
			accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
			predicted_test_array.append(list(accuracy[0]))
			predicted_train_array.append(list(accuracy[1]))
			predicted_test_acc_array.append(list(accuracy[2])) 
			predicted_train_acc_array.append(list(accuracy[3]))

			print(' ')
			if CALCULATE_RETURNS == 'y':
				returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
				ret_pt_tot_train.append(list(returns[0]))
				ret_pt_cor_inc_train.append(list(returns[1]))
				ret_pt_tot_test.append(list(returns[2]))
				ret_pt_cor_inc_test.append(list(returns[3]))
			print ('------------------------------------------')
	c_array = np.asarray(c_array).T
	g_array = np.asarray(g_array).T
	predicted_train_array = np.asarray(predicted_train_array).T
	predicted_train_acc_array = np.asarray(predicted_train_acc_array).T
	predicted_test_acc_array = np.asarray(predicted_test_acc_array).T
	predicted_test_array = np.asarray(predicted_test_array).T
	actual_dist_array = np.asarray(actual_dist_array).T
	out = np.vstack((c_array,g_array,actual_dist_array,predicted_train_array,predicted_test_array,predicted_train_acc_array,predicted_test_acc_array,ret_pt_tot_train,ret_pt_cor_inc_train,ret_pt_tot_test,ret_pt_cor_inc_test))
	#out = out.T
	#header = ['c','gamma','dist_plus_actual','dist_minus_act','pred_plus_train','pred_minus_train','pred_plus_test','pred_minus_test','pred_tain_accuracy_tot','pred_train_acc_plus','pred_train_acc_minus','pre_test_acc_tot','pred_test_acc_plus','pred_test_acc_minus','ret_pt_tot_train','ret_pt_tot_plus','ret_pt_train_minus','ret_pt_cor_train','ret_pt_inc_train','rt_pt_tot_test','rt_pt_plus_test','rt_pt_minus_test','rt_pt_cor_test','ret_pt_inc_test']	
	#header = np.asarray(header)
	#out = np.vstack((header,out))
	np.savetxt("c_gaama.csv", out.T, delimiter=",")	
	return accuracy[2][0], pred_test

def RunRF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	model 				= RandomForestClassifier(n_estimators=301, criterion='gini', max_depth=40, min_samples_split=2, min_samples_leaf=10,
									 min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.00, 
									 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, 
									 warm_start=False, class_weight={1:1,-1:1.15})
	relevant_features 	= FeatureSelection(X_train, Y_train, model, N_FEATURES)
	cv = TimeSeriesSplit(n_splits=3)
#	cv = 3
	min_samples_leaf = [50*i for i in range(1,7)]
	min_samples_split = [i*2 for i in range(1,10)]
	param_grid =  [{'min_samples_leaf':min_samples_leaf, 'min_samples_split':min_samples_split}]
	clf = model_selection.GridSearchCV(model, param_grid, scoring=None, fit_params=None, n_jobs=4, iid=True, 
									refit='best_score_', cv=cv, verbose=0, pre_dispatch='2*n_jobs', 
									error_score='raise', return_train_score='warn')
	clf.fit(X_train, Y_train)
	x = (clf.best_params_ )
	print x
#	exit()
	model.set_params(**x)
	model.fit(X_train, Y_train)
#	model 				= RandomForestClassifier(n_estimators=300, criterion='entropy',random_state = 0)
	X_train_			= X_train[:,relevant_features]
	X_test_				= X_test[:,relevant_features]
	model.fit(X_train_, Y_train)
	pred_test 		= model.predict(X_test_)
	pred_train 		= model.predict(X_train_)
	cnf_mat_test 	= GenerateCnfMatrix(pred_test, Y_test)
	cnf_mat_train 	= GenerateCnfMatrix(pred_train, Y_train)
	actual_dist 	= ComputeDistribution(Y_train, Y_test)	
	accuracy 		= ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist)
#	print np.mean(cross_val_score(model, X_train, Y_train, cv=100))
	if CALCULATE_RETURNS == 'y':
		returns 		= ComputeReturns(Abs_test, Abs_train, pred_test, pred_train, Y_test, Y_train, name)
	print ('------------------------------------------')
	return accuracy[2][0], pred_test

def RunERF(Abs_train, Abs_test, X_train, Y_train, X_test, Y_test, name):
	model 				= ExtraTreesClassifier()
	relevant_features 	= FeatureSelection(X_train, Y_train, model, N_FEATURES)
	X_train_			= X_train[:,relevant_features]
	X_test_				= X_test[:,relevant_features]
	model.fit(X_train_, Y_train)
#	print np.argsort(model.feature_importances_)
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

def SelectFeaturesRF(X_train, Y_train):
	model 				= RandomForestClassifier(n_estimators=301, criterion='gini', max_depth=40, min_samples_split=2, min_samples_leaf=10,
									 min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.00, 
									 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, 
									 warm_start=False, class_weight={1:1,-1:1.15})
	cv 					= TimeSeriesSplit(n_splits=3)
	min_samples_leaf 	= [10*i for i in range(1,10)]
	min_samples_split 	= [i*2 for i in range(1,20)]
	param_grid 			=  [{'min_samples_leaf':min_samples_leaf, 'min_samples_split':min_samples_split}]
	clf 				= model_selection.GridSearchCV(model, param_grid, scoring=None, fit_params=None, n_jobs=-1, iid=True, 
									refit='best_score_', cv=cv, verbose=0, pre_dispatch='2*n_jobs', 
									error_score='raise', return_train_score='warn')
	clf.fit(X_train, Y_train)
	x = (clf.best_params_ )
	model.set_params(**x)
	model.fit(X_train, Y_train)
	return np.argsort(model.feature_importances_)

# call to the main function
MAIN(filename)