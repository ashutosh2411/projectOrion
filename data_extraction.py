
# coding: utf-8

# In[51]:

import pandas as pd
import numpy as np
from sklearn import svm

features = np.genfromtxt("ICICIBANK-I.csv" ,delimiter=','   )

data=[]
for x in features :
    data.append(x[2:])
 
data = np.transpose(data)

tmp= data[3] 
tmp_ = np.hstack((tmp,np.zeros(1)))
tmp = np.hstack((np.zeros(1),tmp))
closing_closing =  tmp_ - tmp
closing_closing = closing_closing[1:-1]

print(closing_closing )
 
tmp= data[0] 
tmp_ = np.hstack((tmp,np.zeros(1)))
tmp = np.hstack((np.zeros(1),tmp))
opening_opening =  tmp_ -tmp
opening_opening =opening_opening[1:-1] 

print(opening_opening )


closing_opening = data[3] -data[0] 
closing_opening = closing_opening[1:]

print(closing_opening)

y = y[1:-1]


 







# In[13]:


classifier = svm.SVC()
classifier.fit(features, )


# In[ ]:



