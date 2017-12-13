import numpy as np 
import sklearn  as sk

data = np.genfromtxt("NIFTY-I.csv" ,delimiter="," , autostrip = True )

def main() :
	
	print(rsi(5))









#calculating RSI for 5 ,10 ,15 days

def rsi(days) :
	rsi_ =[]
	profit = 0 
	loss = 0 

	# calculating for first days(5,10,15)
	for i in data[:,days] :
		if( i[2] - i[5] >0) :
			loss +=i[2] - i[5]
		else :
			profit += i[5] -i[2] 


	for i in range(len(data[days:])) :
		rs = profit/loss
		rsi.append(100-100/(1+rs))
		if(data[i-days][2] > data[i-days][5] ) :
			loss -= data[i-days][2] - data[i-days][5] 
		else :
			profit -= data[i-days][5] - data[i-days][2] 

		if( data[i][2] > data[i][5] ) :
			loss +=data[i][2] - data[i][5] 
		else :
			profit += data[i][5] -data[i][2] 






main()
