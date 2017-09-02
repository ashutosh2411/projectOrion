"""
Program to implement svm on processed data
"""
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import svm

x_csv = [f for f in listdir('../datasets_pro') if isfile(join('../datasets_pro', f)) and f.endswith('.csv')]
for x in x_csv:

	x_address = '../datasets_pro/' + x
	overall = np.genfromtxt(x_address, delimiter = ',')
	np.random.shuffle(overall)										# splitting in train and test randomly
	test, training = overall[:1000,:], overall[1000:,:]
	
	clf = svm.SVC()
	clf.fit(training[:,:6], training[:,7])	

	print clf.predict(test[:,:6]) 