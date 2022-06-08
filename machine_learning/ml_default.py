"""
Author: Alauddin Ahmed (06/06/2022)

"""
 
from __future__ import print_function
import time
import os
import sklearn
import subprocess
from sklearn import datasets
from sklearn import svm
import pandas as pd
import numpy as np


mof = pd.read_csv("Inp_CUS_NON_CUS_All_v1.csv", low_memory=False)

names = list(mof.columns[0:])
mof = mof.values
name = mof[:,0:1]


Size_of_Data_Set = 1000
meta_data_1 = mof[:,0:7]
X_orig= mof[:,4:13]
y_orig = np.ravel(mof[:,13:14])


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error
import scipy.stats as stats


rng = np.random.RandomState(100)


ESTIMATORS = {
            
    "Decision tree":DecisionTreeRegressor(),
    "Linear regression": LinearRegression(),
    

}

X_split = X_orig
y_split = y_orig


X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.25, random_state=1)

X_unseen = X_test
y_unseen = y_test


N = name[:Size_of_Data_Set]
X = X_train[:Size_of_Data_Set]
y = y_train[:Size_of_Data_Set]



model = dict()

kf = KFold(n_splits=10, random_state=None, shuffle=True)

for train_index, test_index in kf.split(X):

   mof_desc_train, mof_desc_test = X[train_index], X[test_index]
   mof_prop_train, mof_prop_test = y[train_index], y[test_index]

   
   for name, estimator in ESTIMATORS.items():
       model[name] = estimator.fit(mof_desc_train, mof_prop_train)
   
y_test_predict = dict()
test_score = dict()
test_r2_score = dict()
test_aue = dict()
test_rmse = dict()
test_kt = dict()
test_ev = dict()
test_med = dict()
test_mape = dict()
rmse = dict()
 
print("name, test_score,test_r2_score, test_aue,test_rmse,test_kt, test_ev,test_med\n")

for name, estimator in ESTIMATORS.items():
   y_test_predict[name] = estimator.predict(X_unseen)
   test_score[name] = model[name].score(X_unseen, y_unseen)
   test_r2_score[name] = r2_score(y_unseen, y_test_predict[name])
   test_aue[name] =  mean_absolute_error(y_unseen, y_test_predict[name])
   test_rmse[name] = mean_squared_error(y_unseen, y_test_predict[name])
   rmse[name] =  test_rmse[name]**(.5)
   test_kt[name] = stats.kendalltau(y_unseen, y_test_predict[name])
   test_ev[name] = explained_variance_score(y_unseen, y_test_predict[name])
   test_med[name] = median_absolute_error(y_unseen, y_test_predict[name])

   print('{}: {}, {}, {}, {}, {}, {}, {}'.format(name, test_score[name],test_r2_score[name], test_aue[name],rmse[name],test_kt[name][0],
                                                 test_ev[name],test_med[name]))

    



