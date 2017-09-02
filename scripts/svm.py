"""
Program to implement svm on processed data
"""
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import svm


def fun(y_index ,test, training):
	#test, training = dataset[:1000,:], dataset[1000:,:]
	
	clf = svm.SVC()
	clf.fit(training[:,:4], training[:,y_index])	
	
	y_hat = clf.predict(test[:,:4])
	y = test[:,y_index]


	#calculating error
	error = 0
	for x in range (len(y_hat)):
		if y_hat[x] != y[x]:
			error = error + 1
	print float(error)/10




x_csv = [f for f in listdir('../datasets_pro') if isfile(join('../datasets_pro', f)) and f.endswith('.csv')]
for x in x_csv:

	x_address = '../datasets_pro/' + x
	overall = np.genfromtxt(x_address, delimiter = ',')
	np.random.shuffle(overall)										# splitting in train and test randomly
	test, training = overall[:1000,:], overall[1000:,:]
	

	y_set = ['opening - opening','closing - closing', 'closing - opening']

	print x
	for i in [5,6,7] :
		print y_set[i-5]
#		print type(overall)
		fun(int(i), test, training)
	print ''




	'''
	#clf = svm.SVC()
	#clf.fit(training[:,:5], training[:,8])	
	#print training[:,:7]
	y_hat = clf.predict(test[:,:5])
	y = test[:,8]
	#print y.shape
	error = 0
	for x in range (len(y_hat)):
		if y_hat[x] != y[x]:
			error = error + 1
	#print np.dot(clf.predict(test[:,:5]) , test[:,7])
	print float(error)/10



	'''