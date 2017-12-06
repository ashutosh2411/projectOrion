# program to calculate indicators using TA-lib

import datetime
import talib as ta
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import svm
import pandas

def main():
	only_csv_files = [f for f in listdir('../datasets_raw') if isfile(join('../datasets_raw', f)) and f.endswith('.csv')]
	in_address = 'ICICIBANK.csv'
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
	o_ = np.hstack((o[1:],0))
	v = v.astype(float)
	y = ycc(c,11) 
	y_hc = yoc(c,h)
	y_lc = yoc(c,l)
	ypast = yccpast(c,11)
	#print ypast.shape
	yoc_cc = yocpast(o,c)
	yoc_oc = np.hstack((yoc_cc[1:],0))
	out = [0]*y.shape[1]
	for i in range(len(y)):
		out = np.vstack((out, y[i]))
		split = calculate_percentile(y[i],2)
		out = np.vstack((out, compute_labels(split, y[i])))
		split = calculate_percentile(y[i],1)
		out = np.vstack((out, compute_labels(split, y[i])))
	t1 = yoc(o,c)
	t2 = compute_labels(calculate_percentile(yoc(o,c),2),yoc(o,c))
	t3 = compute_labels(calculate_percentile(yoc(o,c),1),yoc(o,c))

	out = np.vstack((t1,t2,t3,out[1:]))
	i1 = RSI (c,14)
	
	i2, i3, i4 = MACD (c,6,13,5)
	i5 = WILLR (h,l,c,14)
	i6, i7 = STOCH(h,l,c,5,3,0,3,0)
	i8, i9 = STOCHRSI(c,14,5,3,0)
	i10 = CCI (h,l,c,14)
	i11 = ROC (c,10)
	i12 = OBV (c,v)
	i13 = AD (h,l,c,v)
	i14 = MOM (c,10)
	i15 = SMA (c,10)
	i16 = WMA (c,10)
	i17 = EMA (c,10)
	i18 = TSF (c,10)
	i19, i20, i21 = BBANDS(c,5,2,2,0)
	i22 = TEMA(c,10)
	i23 = ADX (h,l,c,14)
	i24 = MFI (h,l,c,v,14)
	i25 = ATR (h,l,c,14)
	i26 = DIS(c,SMA(c,5))
	i27 = DIS(c,SMA(c,10))
	i28 = DIS(c,WMA(c,5))
	i29 = DIS(c,WMA(c,10))
	i30 = OCRSI(o,c,14)
	#saving indicators to ind.csv
	indicators = np.vstack((i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26,i27,i28,i29,i30))
	np.savetxt("ind.csv", indicators.T, delimiter=",", header = 'RSI,MACD,MACD,MACD,WILLR,STOCH,STOCH,STOCHRSI,STOCHRSI,CCI,ROC,OBV,AD,MOM,SMA,WMA,EMA,TSF,BBANDS,BBANDS,BBANDS,TEMA,ADX,MFI,ATR,DS5,DS10,DW5,DW10,OCRSI')
	
	#saving all the actual and quantized values of actual yoc and ycc
	np.savetxt("out.csv", out.T, delimiter=",")
	
	head = ['date','day','o_yday','h_yday','l_yday','c_yday','v_yday','o_tday','lag_1','lag_2','lag_3','lag_4','lag_5','lag_6','lag_7','lag_8','lag_9','lag_10','yhc','ylc','yoc_past','RSI','MACD','MACDsig','MACDhist','WILLR','slowk','slowd','fastk','fastd','CCI','ROC','OBV','AD','MOM','SMA','WMA','EMA','TSF','BBANDSu','BBANDSm','BBANDSl','TEMA','ADX','MFI','ATR','DS5','DS10','DW5','DW10','OCRSI','yoc_abs','yoc2','yoc1']	
	#print len(head)
	array_oc = np.vstack((date_yoc, day_yoc, o,h,l,c,v,o_,ypast,y_hc,y_lc,yoc_oc,indicators,out[0:3])).T
	array_oc = pandas.DataFrame(np.vstack((head,array_oc)))
	
	head = ['date','day','o_tday','h_tday','l_tday','c_tday','v_tday','lag_1','lag_2','lag_3','lag_4','lag_5','lag_6','lag_7','lag_8','lag_9','lag_10','yhc','ylc','yoc_past','RSI','MACD','MACDsig','MACDhist','WILLR','slowk','slowd','fastk','fastd','CCI','ROC','OBV','AD','MOM','SMA','WMA','EMA','TSF','BBANDSu','BBANDSm','BBANDSl','TEMA','ADX','MFI','ATR','DS5','DS10','DW5','DW10','OCRSI','yoc_abs','yoc2','yoc1','ycc1_abs','ycc1_qnt','ycc1_sgn','ycc2_abs','ycc2_qnt','ycc2_sgn','ycc3_abs','ycc3_qnt','ycc3_sgn','ycc4_abs','ycc4_qnt','ycc4_sgn','ycc5_abs','ycc5_qnt','ycc5_sgn','ycc6_abs','ycc6_qnt','ycc6_sgn','ycc7_abs','ycc7_qnt','ycc7_sgn','ycc8_abs','ycc8_qnt','ycc8_sgn','ycc9_abs','ycc9_qnt','ycc9_sgn','ycc10_abs','ycc10_qnt','ycc10_sgn']	
	array_cc = np.vstack((date_ycc, day_ycc, o,h,l,c,v,ypast,y_hc,y_lc,yoc_cc,indicators,out)).T
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
	out = out[:,]
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
	out = out[:,]
	return out[1:,:]

# returns the split for data. 
def calculate_percentile (data, nDivs):
	data_ = data
	positive = data > 0
	negative = data < 0
	s1 = sorted(data[positive])
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
	split = np.hstack((5000, split,-5000))
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

main()