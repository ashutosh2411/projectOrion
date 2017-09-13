from svmutil import *
import numpy as np
Ytr ,Xtr = svm_read_problem('data.csv')
Yts ,Xts = svm_read_problem('datatset.csv')
prob = svm_problem(Ytr, Xtr)
m = svm_train(prob)
p_label, p_acc, p_val = svm_predict(Yts, Xts , m )
print(Yts)
print(p_label)
