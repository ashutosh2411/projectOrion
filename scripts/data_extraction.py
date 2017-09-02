import numpy as np

def same_attribute_difference (data):
	tmp  = data
	tmp_ = np.hstack((tmp,0))
	tmp  = np.hstack((0,tmp))
	ret  = tmp_ -tmp
	return np.hstack((0,ret[1:-1])) 

def diff_attribute_difference (data1, data2):
	return data2 - data1

features = np.genfromtxt("../datasets_raw/ICICIBANK-I.csv" ,delimiter=','   )
data = features.T[2:-4]

closing_closing = same_attribute_difference(data[3])
print(closing_closing)
 
opening_opening = same_attribute_difference(data[0])	
print(opening_opening)

closing_opening = diff_attribute_difference(data[0], data[3])	
print(closing_opening)

out = np.vstack((closing_closing, opening_opening, closing_opening))
print out.T

np.savetxt("../datasets_pro/svm_ICICIBANK-I.csv", out.T, fmt = '%.4f,%.4f,%.4f')