#!/usr/bin/python
########################################################################################################################
################# ALL INDICATOR LISTS ARE TRIMMED FROM TOP TO ALLIGN IN SAME LENGTH OF 2981#############################
######################################################################################################################## 
import numpy as np 
import sklearn  as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

data = np.genfromtxt("NIFTY-I.csv" ,delimiter="," , autostrip = True )
 
def main() :
    
    #calculating relative strength index
    feature_matrix =[]
    #for i in range(1,4) :
     #   feature_matrix.append( rsi(5*i)) 
    
    Ytr = []
    opening = []
    closing = []
    high    = []
    low     = []
    volume  = []
    for i in range (15, 2996):
    	opening.append(data[i][2])
    	closing.append(data[i-1][5])
    	high.append(data[i-1][3])
    	low.append(data[i-1][4])
    	volume.append(data[i-1][6])
    	if data[i][2] > data[i][5]:
    		Ytr.append(-1)
    	else:
    		Ytr.append(1)


    Ytr = np.asarray(Ytr)

    feature_matrix.append(opening)
    feature_matrix.append(closing)
    feature_matrix.append(high)
    feature_matrix.append(low)
    feature_matrix.append(volume)
    
   
    #calculating stochastic oscillator indicator for last 14 days
    feature_matrix.append(stochastic_osci())
    #feature_matrix.append(rsi(10))
    #calculating william indicator for last 14 days
    feature_matrix.append(william())
    
    #calculating  price_rate_change for last n  days
    feature_matrix.append(price_rate_change(5))

    #calculating on balance volume
    feature_matrix.append(on_balance_volume())

    #calculating simple moving average for n days 
    feature_matrix.append(simple_moving_average(5))
    feature_matrix.append(simple_moving_average(10))
    
    #calculating weighted moving average for n days 
    feature_matrix.append(weighed_moving_average(5))
    feature_matrix.append(weighed_moving_average(10))
    
    #calculating stochastic %d   for n days
    feature_matrix.append(stochastic_d(5))
    
    #calculating exponential moving average for n days
    feature_matrix.append(exponential_moving_aver(10))
    feature_matrix.append(exponential_moving_aver(15))


    feature_matrix = np.array(feature_matrix)
    feature_matrix = feature_matrix.T

    np.savetxt("out.csv", feature_matrix, delimiter=",")


    #clf = svm.SVC()
    
    #clf.fit(feature_matrix[:2000], Ytr[:2000])

    classifier = RandomForestClassifier(n_estimators=80, criterion='gini', max_depth=None,
     					min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
    					 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
     					min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
     					random_state=None, verbose=0, warm_start=False, class_weight=None)
    classifier.fit(feature_matrix[:2000], Ytr[:2000])

    print(classifier.score(feature_matrix[2000:],Ytr[2000:]))
    s = 0
    for i in range(25):
        s = s + rf(feature_matrix,Ytr,i+1)
    print 'avg: '+str(s/25)
   # print(clf.score(feature_matrix[2000:],Ytr[2000:]))

def rf(feature_matrix, Ytr, i):
    classifier = RandomForestClassifier(n_estimators=10*i, criterion='gini', max_depth=None,
                        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                         max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                        min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
                        random_state=None, verbose=0, warm_start=False, class_weight=None)
    classifier.fit(feature_matrix[:2000], Ytr[:2000])

    return(classifier.score(feature_matrix[2000:],Ytr[2000:]))

################################################################################
def rsi(days) :
    rsIndicator =[]
    profit = 0 
    loss = 0 
    # calculating for first days(5,10,15)
    for i in data[:days]  :
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
    sOindicator = sOindicator[:-1]   
    return sOindicator 

##########################################################################
def william() :
    williamIndicator = []
    for i in range(14, len(data)) :
        numerator = max(data[i-14:i,3]) - data[i][5]
        denominator = max(data[i-14:i ,3])-min(data[i-14:i,4])
                                   
        
        williamIndicator.append(numerator/denominator*(-100))
    williamIndicator = williamIndicator[:-1]   
    return williamIndicator

##########################################################################
def price_rate_change(days) :
    indicator = []
    for i in range(days, len(data)) :
        indicator.append((data[i][5] - data[i-days +1 ][5])/data[i-days +1][5] )
    if days == 5 :
    	  indicator = indicator[9:-1]
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
    indicator = indicator[14:-1]    
    return indicator 

##########################################################################
def simple_moving_average(days) :
    indicator = []
    for i in range(days, len(data)) :
        
        indicator.append(sum(data[i-days +1 :i,5])/days)
    if days == 5 :
    	  indicator = indicator[9:-1]
    elif days == 10 :
    	  indicator = indicator[4:-1]   
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
    	  indicator = indicator[9:-1]
    elif days == 10 :
    	  indicator = indicator[4:-1]    
    return indicator

##########################################################################

def stochastic_d(days):
    li = stochastic_osci() 
    indicator = []
    for i in range(days,len(data)) :
        indicator.append(sum(li[i-days+1: i ])/days)
    indicator = indicator[9:-1]
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
    	  indicator = indicator[4:-1]
    return indicator 




main()





