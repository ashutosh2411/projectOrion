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

def process_file (data, out_address):
# process file to get the closing_closing, opening_opening, closing_opening in a matrix
	feature_mat = data.T[2:-3]
#	print (feature_mat.shape)
	feature_mat = np.vstack((feature_mat[:,15:], 
					stochastic_osci(),
					william(),
					price_rate_change(5),
					on_balance_volume(),
					simple_moving_average(5),
					simple_moving_average(10),
					weighed_moving_average(5),
					weighed_moving_average(10),
					stochastic_d(5),
					exponential_moving_aver(10),
					exponential_moving_aver(15) ))

	closing_closing = np.sign(same_attribute_difference(data.T[2+3]))[15:]
#	print closing_closing
	opening_opening = np.sign(same_attribute_difference(data.T[2+0]))[15:]
#	print opening_opening
	closing_opening = np.sign(diff_attribute_difference(data.T[2+0], data.T[2+3]))[15:]
#	print closing_opening

	out = np.vstack((closing_closing, opening_opening, closing_opening))
	overall = np.hstack((feature_mat.T, out.T))
	head = 'open,high,low,clos,volm,sto,will,prate,bvol,sma1,sma2,wma1,wma2,std,ema1,ema2,ycc,yoo,yoc'
	#overall = np.vstack((head,overall))  
	np.savetxt(out_address, overall, header = head, fmt = '%.2f,\t'*7+'%.8f,\t'+'%.2f,\t'*8+'%d,\t%d,\t%d', comments='')
	#np.savetxt(out_address[:-4]+'_out.csv', out.T, header = 'ycc,y00,yoc', fmt = '%d,\t%d,\t%d')

def rsi(days) :
	rsIndicator =[]
	profit = 0 
	loss = 0 
	# calculating for first days(5,10,15)
	for i in data  :
		if( i[2] - i[5] >0) :
			loss +=i[2] - i[5]
		else :
			profit += i[5] -i[2] 


	for i in range(len(data[days:])) :
		rs = profit/loss
		rsIndicator.append(100-100/(1+rs))
		if(data[i-days][2] > data[i-days][5] ) :
			loss -= data[i-days][2] - data[i-days][5] 
		else :
			profit -= data[i-days][5] - data[i-days][2] 

		if( data[i][2] > data[i][5] ) :
			loss +=data[i][2] - data[i][5] 
		else :
			profit += data[i][5] -data[i][2] 
	if days == 5 :
		  rsIndicator = rsIndicator[10:]
	elif days == 10 :
		rsIndicator = rsIndicator[5:]    
	return rsIndicator
	
	
##########################################################################
def stochastic_osci() :
	sOindicator = []
	for i in range(14, len(data)) :
		numerator = data[i][5]-min(data[i-14:i,4])
		denominator = max(data[i-14:i ,3])-min(data[i-14:i,4])
								   
		
		sOindicator.append(numerator/denominator*100)
	sOindicator = sOindicator[1:]   
	return sOindicator 

##########################################################################
def william() :
	williamIndicator = []
	for i in range(14, len(data)) :
		numerator = max(data[i-14:i,3]) - data[i][5]
		denominator = max(data[i-14:i ,3])-min(data[i-14:i,4])
								   
		
		williamIndicator.append(numerator/denominator*(-100))
	williamIndicator = williamIndicator[1:]   
	return williamIndicator

##########################################################################
def price_rate_change(days) :
	indicator = []
	for i in range(days, len(data)) :
		indicator.append((data[i][5] - data[i-days +1 ][5])/data[i-days +1][5] )
	if days == 5 :
		  indicator = indicator[10:]
	return indicator 


##########################################################################
def on_balance_volume() :
	indicator = [0]
	obv = 0
	for i in range(1, len(data)) :
		if(data[i][5] > data[i-1][5]) :
			obv = obv + data[i][6]
			indicator.append(obv)
		elif (data[i][5] < data[i-1][5]) :
			obv = obv - data[i][6]
			indicator.append(obv)
		else :
			indicator.append(obv)
	indicator = indicator[15:]    
	return indicator 

##########################################################################
def simple_moving_average(days) :
	indicator = []
	for i in range(days, len(data)) :
		
		indicator.append(sum(data[i-days +1 :i,5])/days)
	if days == 5 :
		  indicator = indicator[10:]
	elif days == 10 :
		  indicator = indicator[5:]   
	return indicator



##########################################################################
def weighed_moving_average(days) :
	indicator = []
	li =[]
	for i in range(days-1):
		li.append(days-i)
	li = np.asarray(li)
	for i in range(days, len(data)) :
		x = np.dot(data[i-days + 1:i,5],li)
		indicator.append(float(x)/sum(li))
	if days == 5 :
		  indicator = indicator[10:]
	elif days == 10 :
		  indicator = indicator[5:]    
	return indicator

##########################################################################

def stochastic_d(days):
	li = stochastic_osci() 
	indicator = []
	for i in range(days,len(data)) :
		indicator.append(sum(li[i-days+1: i])/days)
	indicator = indicator[10:]
	return indicator 

##########################################################################

def exponential_moving_aver(days):
	indicator = []
	alpha = 2/float(1+days)
	li =[ alpha ]
	for i in range(1,days-1) :
		li.append(pow((1-alpha),i))    
	for i in range(days,len(data)) :    
		indicator.append( np.dot(li , data[i-days+1:i,5])/sum(li))
	if days == 10 :
		  indicator = indicator[5:]
	return indicator 

x_csv = [f for f in listdir('../datasets_raw') if isfile(join('../datasets_raw', f)) and f.endswith('.csv') and f.startswith('svm_')]
for x in x_csv:
	x_address = '../datasets_pro/' + x
	data = np.genfromtxt(x_address ,delimiter=",")
	test, training = overall[:1000,:], overall[1000:,:]
	



only_csv_files = [f for f in listdir('../datasets_raw') if isfile(join('../datasets_raw', f)) and f.endswith('.csv')]
for x in only_csv_files:
	
	in_address  = '../datasets_raw/' + x
	out_address = '../datasets_pro/indi_' + x[:-4] + '.csv'
	data = np.genfromtxt(in_address,delimiter=',')
	process_file (data, out_address)