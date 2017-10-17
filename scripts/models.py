import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import pandas

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

in_columns  = ['open', 'high', 'low', 'clos', 'volm', 'sto', 'will', 'prate', 'bvol', 'sma1', 'sma2', 'wma1', 'wma2', 'std', 'ema1', 'ema2']
out_columns = ['ycc']

file = '../datasets_pro/indi_NIFTY-I.csv'

dataset = pandas.read_csv(file , usecols = in_columns+out_columns)

dataset_ = dataset.sample(frac=1)
array = dataset_.values
X = array[:,0:-1]
Y = array[:,-1]
validation_size = 0.1
seed = 9
scoring = 'accuracy'

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

models = []
models.append(('LR ', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF ', DecisionTreeClassifier()))
models.append(('NB ', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

print '--------------------'

lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict (X_validation)
print 'LR : ' + str(accuracy_score(Y_validation, predictions))

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict (X_validation)
print 'LDA: ' +str(accuracy_score(Y_validation, predictions))

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict (X_validation)
print 'KNN: '+str(accuracy_score(Y_validation, predictions))

rf = DecisionTreeClassifier()
rf.fit(X_train, Y_train)
predictions = rf.predict (X_validation)
print 'RF : ' +str(accuracy_score(Y_validation, predictions))

nb = GaussianNB()
nb.fit(X_train, Y_train)
predictions = nb.predict (X_validation)
print 'NB : '+str(accuracy_score(Y_validation, predictions))	

svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict (X_validation)
print 'SVM: '+str(accuracy_score(Y_validation, predictions))	

print '--------------------'
