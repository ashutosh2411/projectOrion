# program to calculate indicators using TA-lib

import datetime
import talib as ta
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import svm
import pandas

def main(var):
	in_address = 'NIFTY.csv'
	dataset = pandas.read_csv(in_address)
	data = dataset.values
	date_ycc = data[:,0]
	date_yoc = np.hstack((data[1:,0],'11-11-1911'))
	day_ycc = day_to_date(date_ycc)
	day_yoc = day_to_date(date_yoc)

	data = data[:,[2,3,4,5,6]]
	##print data.shape
	o = np.array(list(data[:,0]))
	h = np.array(list(data[:,1]))
	l = np.array(list(data[:,2]))
	c = np.array(list(data[:,3]))
	v = np.array(list(data[:,4]))
	o_ = np.hstack((o[1:],np.nan))
	c_ = np.hstack((c[1:],np.nan))
	v = v.astype(float)
	tam = range(10)+[14,19,29]
	y = ycc(c,31)[tam]
	y_hc = yoc(c,h)
	y_lc = yoc(c,l)
	ypast = yccpast(c,11)
	#print ypast.shape
	yoc_cc = yocpast(o,c)
	yoc_oc = np.hstack((yoc_cc[1:],0))
	out = [0]*y.shape[1]
	for i in range(len(y)):
		out = np.vstack((out, y[i]))
		split = calculate_percentile(y[i],1)
		out = np.vstack((out, compute_labels(split, y[i])))
		split = calculate_percentile(y[i],2)
		out = np.vstack((out, compute_labels(split, y[i])))
		split = calculate_percentile(y[i],3)
		out = np.vstack((out, compute_labels(split, y[i])))
	t1 = np.hstack((yoc(o,c)[1:],np.nan))
	t2 = np.hstack((compute_labels(calculate_percentile(yoc(o,c),2),yoc(o,c))[1:],np.nan))
	t3 = np.hstack((compute_labels(calculate_percentile(yoc(o,c),1),yoc(o,c))[1:],np.nan))

	i = []
	names = []
	out = np.vstack((t1,t2,t3,out[1:]))
	i.append( MACD (c,12,26,9)[0])
	names.append('MACD')
	i.append( MACD (c,12,26,9)[1])
	names.append('MACDsig')
	i.append( MACD (c,12,26,9)[2])
	names.append('MACDhist')
	i.append( STOCH(h,l,c,5,3,0,3,0)[0])
	names.append('slowk')
	i.append( STOCH(h,l,c,5,3,0,3,0)[1])
	names.append('slowd')
	i.append( STOCHRSI(c,14,5,3,0)[0])
	names.append('fastk')
	i.append( STOCHRSI(c,14,5,3,0)[1])
	names.append('fastd')
	i.append( OBV (c,v))
	names.append('OBV')
	i.append( AD (h,l,c,v))
	names.append('AD')
	i.append( BBANDS(c,20,2,2,0)[0])
	names.append('Bband u')
	i.append( BBANDS(c,20,2,2,0)[1])
	names.append('bband m')
	i.append( BBANDS(c,20,2,2,0)[2])
	names.append('bband l')
	for var in range(1,5):
		i.append( RSI (c,var*5))
		names.append('rsi'+str(var))
		i.append( WILLR (h,l,c,var*5))
		names.append('willr'+str(var))
		i.append( CCI (h,l,c,var*10))
		names.append('cci'+str(var))
		i.append( ROC (c,var*5))
		names.append('roc'+str(var))
		i.append( MOM (c,var*5))
		names.append('mom'+str(var))
		i.append( SMA (c,var*5))
		names.append('sma'+str(var))
		i.append( WMA (c,var*5))
		names.append('wma'+str(var))
		i.append( EMA (c,var*5))
		names.append('ema'+str(var))
		i.append( TSF (c,var*5))
		names.append('tsf'+str(var))
		i.append( TEMA(c,var*5))
		names.append('tema'+str(var))
		i.append( ADX (h,l,c,var*5))
		names.append('adx'+str(var))
		i.append( MFI (h,l,c,v,var*5))
		names.append('mfi'+str(var))
		i.append( ATR (h,l,c,var*5))
		names.append('atr'+str(var))
		i.append( DIS(c,SMA(c,var*5)))
		names.append('disS'+str(var))
		i.append( DIS(c,WMA(c,var*5)))
		names.append('disW'+str(var))
		i.append( OCRSI(o,c,var*5))
		names.append('ocrsi'+str(var))
	for var in range(1,3):
		i.append( RSI (c,var*7))
		names.append('rsi7'+str(var))
		i.append( WILLR (h,l,c,var*7))
		names.append('will7'+str(var))
		i.append( ADX (h,l,c,var*7))
		names.append('adx7'+str(var))
		i.append( MFI (h,l,c,v,var*7))
		names.append('mfi7'+str(var))
		i.append( ATR (h,l,c,var*7))
		names.append('atr7'+str(var))
		i.append( OCRSI(o,c,var*7))
		names.append('ocrsi7'+str(var))
#saving indicators to ind.csv
	indicators = np.asarray(i)
	np.savetxt("ind.csv", indicators, delimiter=",")
	
	#saving all the actual and quantized values of actual yoc and ycc
	np.savetxt("out.csv", out.T, delimiter=",")
	
	head = ['date','day','o_yday','h_yday','l_yday','c_yday','v_yday','o_tday','c_tday','lag_1','lag_2','lag_3','lag_4','lag_5','lag_6','lag_7',
		'lag_8','lag_9','lag_10','yhc','ylc','yoc_past']+names+['yoc_abs','yoc_qnt','yoc_sgn']	
	array_oc = np.vstack((date_yoc, day_yoc, o,h,l,c,v,o_,c_,ypast,y_hc,y_lc,yoc_oc,indicators,out[0:3])).T
	array_oc = pandas.DataFrame(np.vstack((head,array_oc)))
	out_names = ['ycc1_abs','ycc1_sign','ycc1_qnt2','ycc1_qnt3','ycc2_abs','ycc2_sign','ycc2_qnt2','ycc2_qnt3','ycc3_abs','ycc3_sign','ycc3_qnt2','ycc3_qnt3','ycc4_abs','ycc4_sign','ycc4_qnt2','ycc4_qnt3','ycc5_abs','ycc5_sign','ycc5_qnt2','ycc5_qnt3','ycc6_abs','ycc6_sign','ycc6_qnt2','ycc6_qnt3','ycc7_abs','ycc7_sign','ycc7_qnt2','ycc7_qnt3','ycc8_abs','ycc8_sign','ycc8_qnt2','ycc8_qnt3','ycc9_abs','ycc9_sign','ycc9_qnt2','ycc9_qnt3','ycc10_abs','ycc10_sign','ycc10_qnt2','ycc10_qnt3','ycc15_abs','ycc15_sign','ycc15_qnt2','ycc15_qnt3','ycc20_abs','ycc20_sign','ycc20_qnt2','ycc20_qnt3','ycc30_abs','ycc30_sign','ycc30_qnt2','ycc30_qnt3']
	head = ['date','day','o_tday','h_tday','l_tday','c_tday','v_tday','yoc_abs','lag_1','lag_2','lag_3','lag_4','lag_5','lag_6','lag_7','lag_8'
		,'lag_9','lag_10','yhc','ylc','yoc_past']+names+out_names
	array_cc = np.vstack((date_ycc, day_ycc, o,h,l,c,v,yoc(o,c),ypast,y_hc,y_lc,yoc_cc,indicators,out[3:])).T
	array_cc = pandas.DataFrame(np.vstack((head,array_cc)))

	array_cc.to_csv('ycc.csv',index = False)
	array_oc.to_csv('yoc.csv',index = False)

#converts list of days to list of dates
def day_to_date(date):
	day = []
	for i in date:	
		day.append(datetime.datetime.strptime(i, "%d-%m-%Y").strftime('%A'))
	return day

#calculates the difference between today's opening and yesterday's closing
def yocpast (opn, close):
	close = np.hstack((np.nan,close))
	opn = np.hstack((opn,np.nan))
	return np.divide(np.subtract(opn, close)[:-1],close[:-1])

# calculates opening difference of 
# (ith day in future) - (today)
def yoc (opn, close):
	return np.divide(np.subtract(close, opn),opn)

# calculates closing difference of 
# (ith day in future) - (today)
def ycc (data, day_range):
	nextData = data
	out = data
	length = len(data)
	for i in range (1,day_range):
		data = np.hstack((np.nan,data))
		nextData = np.hstack((nextData,np.nan))
		pred = np.divide((nextData-data),data)
		out = np.vstack((out,pred[i:]))
	return out[1:,:]

def yccpast (data, day_range):
	prevData = data
	out = data
	length = len(data)
	for i in range (1,day_range):
		data = np.hstack((data,np.nan))
		prevData = np.hstack((np.nan,prevData))
		pred = np.divide((data - prevData),prevData)
		out = np.vstack((out,pred[:-i]))
	return out[1:,:]

# returns the split for data. 
def calculate_percentile (data, nDivs):
	data_ = data
	positive = data > 0
	negative = data < 0
	s1 = sorted(data[positive],reverse = True)
	s2 = sorted(data[negative],reverse = True)
	pos = []
	neg = []
	for i in range (1, nDivs):
		pos.append(s1[(len(s1)/nDivs)*i])
		neg.append(s2[(len(s2)/nDivs)*i])
	return np.hstack((pos,0,neg))
	
def process_file (data, out_address):
# process file to get the date closing_closing, closing_opening in a matrix
	feature_mat = data.T[0,2:-3]

# Compute the labels. 
def compute_labels (split, data):
	out = [0]*len(data)
	split = np.hstack((500000, split,-500000))
	label = len(split) /2
	for i in range(1,len(split)):
		for j in range(len(data)):
			if data[j] == 0:
				out[j] = 0
			elif data[j] > split[i] and data[j] <= split[i-1]:
				out[j] = label
		label = label - 1
		if label == 0:
			label = label -1
	return out

def OCRSI (opn, close, day_range):
	gain_loss = close - opn
	gain=[]
	ret =[]
	loss=[]
	for i in gain_loss:
		if i > 0:
			gain.append(i)
			loss.append(0)
		else:
			gain.append(0)
			loss.append(-i)
	ga = sum(gain[:day_range])/day_range
	la = sum(loss[:day_range])/day_range
	for i in range(len(gain)-day_range):
		ga = (ga*(day_range-1)+gain[i])/day_range
		la = (la*(day_range-1)+loss[i])/day_range
		ret.append(100-100/(1+ga/la))
	return [np.nan]*day_range+ret

def DIS (close, ma):
	return np.divide(close , ma)

# Suggests the overbrought and oversold market signals
# 100 - (100 / (1 + (sum_gains_n_days / sum_loss_n_days)))
def RSI (close, days):
	# RELATIVE STRENGTH INDEX
	real_ = ta.RSI(np.array(close),days)
	return real_

# Uses different EMAs to signal buy and sell
# EMA_fast - EMA_slow
def MACD (close, fastperiod, slowperiod, signalperiod):
	# MOVING AVERAGE CONVERGENCE DIVERGENCE
	macd, macdsignal, macdhist = ta.MACD(np.array(close), fastperiod, slowperiod, signalperiod)
	return macd, macdsignal, macdhist

# Determines where today's closing price fell within the range on past n days transaction
# (highest-closed)/(highest-lowest)*100
def WILLR (high, low, close, timeperiod = 14):
	# WILLIAM'S %R
	real_ = ta.WILLR(high, low, close, timeperiod)
	return real_

# The general theory serving as the foundation for this indicator is that in a 
# market trending upward, prices will close near the high, and in a market trending downward, 
# prices close near the low. 
# %K = 100(C - L14)/(H14 - L14)
# %D = 3-period moving average of %K
def STOCH (high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
	# STOCHASTIC %k, %d
	slowk, slowd = ta.STOCH(high, low, close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)
	return slowd, slowk

# gives an idea of whether the current RSI value is overbought or oversold
# RSI over STOCH values
def STOCHRSI (close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
	# STOCHASTIC RSI
	fastk, fastd = ta.STOCHRSI(close, timeperiod, fastk_period, fastd_period, fastd_matype)
	return fastk, fastd

# Identifies cyclic turns in stock prices
# (Price - SMA) / (0.015 * NormalDeviation)`
def CCI (high, low, close, timeperiod=14):
	# COMMODITY CHANNEL INDEX
	real_ = ta.CCI(high, low, close, timeperiod)
	return real_

# Rate of change relative to previous interval
# (Price(t)/Price(t-n))*100
def ROC (close, timeperiod=10):
	# RATE OF CHANGE
	real_ = ta.ROC(close, timeperiod)
	return real_

# Relates trading volume to price change
# OBV(t)=OBV(t-1)+/-Volume(t)
def OBV (close, volume):
	print volume
	# ON BALANCE VOLUME
	real_ = ta.OBV(close, volume)
	return real_

# highlights the momentum implied by the accumulation/distribution line. 
# EMA(nfast of AD line) - EMA(nslow of AD line)
def AD (high, low, close, volume):
	# CHAIKIN'S A/D OSCILLATOR
	real_ = ta.AD(high, low, close, volume)
	return real_

# Momentum for n days
# Close today + Close n_days
def MOM (close, timeperiod=10):
	# N DAYS MOMENTUM
	real_ = ta.MOM(close, timeperiod)
	return real_

# Average of prices for last n days
def SMA (close, timeperiod=10):
	# SIMPLE MOVING AVERAGE
	real_ = ta.SMA(close, timeperiod)
	return real_

# Weighted average of prices for last n days
def WMA (close, timeperiod=10):
	# WEIGHTED MOVING AVERAGE
	real_ = ta.WMA(close, timeperiod)
	return real_

# Exponential average prices for last n days
def EMA (close, timeperiod=10):
	# EXPONENTIAL MOVING AVERAGE
	real_ = ta.EMA(close, timeperiod)
	return real_

# Calculates the Linear regression of n days price
def TSF (close, timeperiod=10):
	# TIME SERIES FORECAST
	real_ = ta.TSF(close, timeperiod)
	return real_

# when the markets become more volatile, the bands widen; during less volatile periods, the bands contract.
def BBANDS (close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
	# BOLLINGER'S BANDS
	upperband, middleband, lowerband = ta.BBANDS(close, timeperiod, nbdevup, nbdevdn, matype)
	return upperband, middleband, lowerband

# Triple exponential average. Smoothens insignificant movements
def TEMA (close, timeperiod=10):
	# TRIPLE EXPONENTIAL MOVING AVERAGE
	real_ = ta.TEMA(close, timeperiod)
	return real_

# Trend strength indicator
def ADX (high, low, close, timeperiod=14):
	# AVERAGE DIRECTION MOVING INDEX
	real_ = ta.ADX(high, low, close, timeperiod)
	return real_

# Relates typical price with volume
def MFI (high, low, close, volume, timeperiod=14):
	# MONEY FLOW INDEX
	real_ = ta.MFI(high, low, close, volume, timeperiod)
	return real_

# Shows volatality of market
def ATR (high, low, close, timeperiod=14):
	# AVERAGE TRUE RANGE
	real_ = ta.ATR(high, low, close, timeperiod=14)
	return real_

main(1)