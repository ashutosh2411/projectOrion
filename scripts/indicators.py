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
	#print data.shape
	o = np.array(list(data[:,0]))
	h = np.array(list(data[:,1]))
	l = np.array(list(data[:,2]))
	c = np.array(list(data[:,3]))
	v = np.array(list(data[:,4]))
	o_ = np.hstack((o[1:],0))
	v = v.astype(float)
	y = ycc(c,11) 
	ypast = yccpast(c,11)
	print ypast.shape
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
	i2, i3, i4 = MACD (c,12,26,9)
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
	
	#saving indicators to ind.csv
	indicators = np.vstack((i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25 ))
	np.savetxt("ind.csv", indicators.T, delimiter=",", header = 'RSI,MACD,MACD,MACD,WILLR,STOCH,STOCH,STOCHRSI,STOCHRSI,CCI,ROC,OBV,AD,MOM,SMA,WMA,EMA,TSF,BBANDS,BBANDS,BBANDS,TEMA,ADX,MFI,ATR')
	
	#saving all the actual and quantized values of actual yoc and ycc
	np.savetxt("out.csv", out.T, delimiter=",")
	
	array_oc = np.vstack((date_yoc, day_yoc, o,h,l,c,v,o_,ypast,yoc_oc,indicators,out)).T
	array_oc = pandas.DataFrame(array_oc)
	
	array_cc = np.vstack((date_ycc, day_ycc, o,h,l,c,v,ypast,yoc_cc,indicators,out)).T
	array_cc = pandas.DataFrame(array_cc)
	
	head = ['0 : date','1 : day','2 : o','3 : h','4 : l','5 : c','6 : v','7 : l1','8 : l2','9 : l3','10 : l4','11 : l5','12 : l6','13 : l7','14 : l8','15 : l9','16 : l10','17 : yoc_past','18 : RSI','19 : MACD','20 : MACDsig','21 : MACDhist','22 : WILLR','23 : slowk','24 : slowd','25 : fastk','26 : fastd','27 : CCI','28 : ROC','29 : OBV','30 : AD','31 : MOM','32 : SMA','33 : WMA','34 : EMA','35 : TSF','36 : BBANDSu','37 : BBANDSm','38 : BBANDSl','39 : TEMA','40 : ADX','41 : MFI','42 : ATR','43 : yoc_actual','44 : yoc2','45 : yoc1','46 : ycc_a','47 : ycc2','48 : ycc1','49 : ycc_a','50 : ycc2','51 : ycc1','52 : ycc_a','53 : ycc2','54 : ycc1','55 : ycc_a','56 : ycc2','57 : ycc1','58 : ycc_a','59 : ycc2','60 : ycc1','61 : ycc_a','62 : ycc2','63 : ycc1','64 : ycc_a','65 : ycc2','66 : ycc1','67 : ycc_a','68 : ycc2','69 : ycc1','70 : ycc_a','71 : ycc2','72 : ycc1','73 : ycc_a','74 : ycc2','75 : ycc10']	#print len(head)
	array_cc.to_csv('ycc.csv',header=head,index = False)
	
	head = ['0 : date','1 : day','2 : o','3 : h','4 : l','5 : c','6 : v','7 : o_','8 : l1','9 : l2','10 : l3','11 : l4','12 : l5','13 : l6','14 : l7','15 : l8','16 : l9','17 : l10','18 : yoc_past','19 : RSI','20 : MACD','21 : MACDsig','22 : MACDhist','23 : WILLR','24 : slowk','25 : slowd','26 : fastk','27 : fastd','28 : CCI','29 : ROC','30 : OBV','31 : AD','32 : MOM','33 : SMA','34 : WMA','35 : EMA','36 : TSF','37 : BBANDSu','38 : BBANDSm','39 : BBANDSl','40 : TEMA','41 : ADX','42 : MFI','43 : ATR','44 : yoc_actual','45 : yoc2','46 : yoc1','47 : ycc_a','48 : ycc2','49 : ycc1','50 : ycc_a','51 : ycc2','52 : ycc1','53 : ycc_a','54 : ycc2','55 : ycc1','56 : ycc_a','57 : ycc2','58 : ycc1','59 : ycc_a','60 : ycc2','61 : ycc1','62 : ycc_a','63 : ycc2','64 : ycc1','65 : ycc_a','66 : ycc2','67 : ycc1','68 : ycc_a','69 : ycc2','70 : ycc1','71 : ycc_a','72 : ycc2','73 : ycc1','74 : ycc_a','75 : ycc2','76 : ycc10']	
	array_oc.to_csv('yoc.csv', header=head,index = False)

#converts list of days to list of dates
def day_to_date(date):
	day = []
	for i in date:	
		day.append(datetime.datetime.strptime(i, "%d-%m-%Y").strftime('%A'))
	return day

#calculates the difference between today's opening and yesterday's closing
def yocpast (opn, close):
	close = np.hstack((0,close))
	opn = np.hstack((opn,0))
	return np.subtract(opn, close)[1:]

# calculates opening difference of 
# (ith day in future) - (today)
def yoc (opn, close):
	return np.subtract(close, opn)

# calculates closing difference of 
# (ith day in future) - (today)
def ycc (data, day_range):
	nextData = data
	out = data
	length = len(data)
	for i in range (1,day_range):
		data = np.hstack((0,data))
		nextData = np.hstack((nextData,0))
		pred = nextData-data
		out = np.vstack((out,pred[i:]))
	out = out[:,]
	return out[1:,:]

def yccpast (data, day_range):
	prevData = data
	out = data
	length = len(data)
	for i in range (1,day_range):
		data = np.hstack((data,0))
		prevData = np.hstack((0,prevData))
		pred = data - prevData
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