"""
Program to process .csv files from folder "../datasets_raw/" to "../datasets_pro/"
"""

import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import svm

def same_attribute_difference (data):
# current day's value - previous days value of an attribute
# first entry is 0
	tmp  = data
	tmp_ = np.hstack((tmp,0))
	tmp  = np.hstack((0,tmp))
	ret  = tmp_ -tmp
	return np.hstack((0,ret[1:-1])) 

def diff_attribute_difference (data1, data2):
# current day's attribute1 - attribute2 values
	return data2 - data1

def process_file (in_address, out_address):
# process file to get the closing_closing, opening_opening, closing_opening in a matrix
	features = np.genfromtxt(in_address,delimiter=',')
	data = features.T[2:-3]
	
	closing_closing = np.sign(same_attribute_difference(data[3]))
	opening_opening = np.sign(same_attribute_difference(data[0]))
	closing_opening = np.sign(diff_attribute_difference(data[0], data[3]))

	out = np.vstack((closing_closing, opening_opening, closing_opening))
	
	t = np.asarray(range(1,data.shape[1]+1))[np.newaxis]
	print data.T.shape
	print t.T.shape
	print out.T.shape

	overall   = np.hstack((t.T, data.T, out.T))

	np.savetxt(out_address, overall, fmt = '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f')


only_csv_files = [f for f in listdir('../datasets_raw') if isfile(join('../datasets_raw', f)) and f.endswith('.csv')]
for x in only_csv_files:
	
	in_address  = '../datasets_raw/' + x
	out_address = '../datasets_pro/svm_' + x[:-4] + '.csv'
	
	process_file (in_address, out_address)