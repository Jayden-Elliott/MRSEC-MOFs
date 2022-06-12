from __future__ import print_function
import time
import os
import sys
import sklearn
import subprocess
from sklearn import datasets
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

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

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# read input csv to pandas dataframe
def main(filename):
    mof = pd.read_csv(filename, low_memory=False)

    # replace missing values with median
    for i in range(4, 178):
        mof.iloc[:, i] = mof.iloc[:, i].replace(0, mof.iloc[:, i].median())

    # convert dataframe to numpy array and remove column names
    names = list(mof.columns[0:])
    mof = mof.values
    name = mof[:, 0]

    # create X and y arrays
    Size_of_Data_Set = len(mof)
    meta_data_1 = mof[:, 0:3]
    X_orig = []
    features = ['f-chi-0-all', 'f-chi-1-all', 'f-chi-2-all', 'f-chi-3-all', 'f-Z-0-all', 'f-Z-1-all', 'f-Z-2-all', 'f-Z-3-all', 'f-I-0-all', 'f-I-1-all', 'f-I-2-all', 'f-I-3-all', 'f-T-0-all', 'f-T-1-all', 'f-T-2-all', 'f-T-3-all', 'f-S-0-all', 'f-S-1-all', 'f-S-2-all', 'f-S-3-all', 'mc-chi-0-all', 'mc-chi-1-all', 'mc-chi-2-all', 'mc-chi-3-all', 'mc-Z-0-all', 'mc-Z-1-all', 'mc-Z-2-all', 'mc-Z-3-all', 'mc-I-1-all', 'mc-I-2-all', 'mc-I-3-all', 'mc-T-0-all', 'mc-T-1-all', 'mc-T-2-all', 'mc-T-3-all', 'mc-S-0-all', 'mc-S-1-all', 'mc-S-2-all', 'mc-S-3-all', 'D_mc-chi-1-all', 'D_mc-chi-2-all', 'D_mc-chi-3-all', 'D_mc-Z-1-all', 'D_mc-Z-2-all', 'D_mc-Z-3-all', 'D_mc-T-1-all', 'D_mc-T-2-all', 'D_mc-T-3-all', 'D_mc-S-1-all', 'D_mc-S-2-all', 'D_mc-S-3-all', 'f-lig-chi-0', 'f-lig-chi-1', 'f-lig-chi-2', 'f-lig-chi-3', 'f-lig-Z-0', 'f-lig-Z-1', 'f-lig-Z-2', 'f-lig-Z-3', 'f-lig-I-0', 'f-lig-I-1', 'f-lig-I-2', 'f-lig-I-3', 'f-lig-T-0', 'f-lig-T-1', 'f-lig-T-2', 'f-lig-T-3', 'f-lig-S-0', 'f-lig-S-1', 'f-lig-S-2', 'f-lig-S-3', 'lc-chi-0-all', 'lc-chi-1-all', 'lc-chi-2-all', 'lc-chi-3-all', 'lc-Z-0-all', 'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-I-1-all', 'lc-I-2-all', 'lc-I-3-all', 'lc-T-0-all', 'lc-T-1-all', 'lc-T-2-all', 'lc-T-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all', 'lc-S-3-all', 'D_lc-chi-1-all', 'D_lc-chi-2-all', 'D_lc-chi-3-all', 'D_lc-Z-1-all', 'D_lc-Z-2-all', 'D_lc-Z-3-all', 'D_lc-T-1-all', 'D_lc-T-2-all', 'D_lc-T-3-all', 'D_lc-S-1-all', 'D_lc-S-2-all', 'D_lc-S-3-all', 'func-chi-0-all', 'func-chi-1-all', 'func-chi-2-all', 'func-chi-3-all', 'func-Z-0-all', 'func-Z-1-all', 'func-Z-2-all', 'func-Z-3-all', 'func-I-0-all', 'func-I-1-all', 'func-I-2-all', 'func-I-3-all', 'func-T-0-all', 'func-T-1-all', 'func-T-2-all', 'func-T-3-all', 'func-S-0-all', 'func-S-1-all', 'func-S-2-all', 'func-S-3-all', 'D_func-chi-1-all', 'D_func-chi-2-all', 'D_func-chi-3-all', 'D_func-Z-1-all', 'D_func-Z-2-all', 'D_func-Z-3-all', 'D_func-T-1-all', 'D_func-T-2-all', 'D_func-T-3-all', 'D_func-S-1-all', 'D_func-S-2-all', 'D_func-S-3-all', 'Df', 'Di', 'Dif', 'GPOAV', 'GPONAV', 'GPOV', 'GSA', 'POAV', 'POAV_vol_frac', 'PONAV', 'PONAV_vol_frac', 'VPOV', 'VSA', 'rho']
    feature_indices = []
    for f in features:
        feature_indices.append(names.index(f))
    X_orig = mof[:, feature_indices]
    y_orig = np.ravel(mof[:, 3])

    rng = np.random.RandomState(100)

    # define ml models to be used
    ESTIMATORS = {
        'decision_tree': DecisionTreeRegressor(),
        'linear_regression': LinearRegression(),
    }

    # split data into train and test groups
    X_split = X_orig
    y_split = y_orig
    X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.20, random_state=1)
    X_unseen = X_test
    y_unseen = y_test

    N = name[:Size_of_Data_Set]
    X = X_train[:Size_of_Data_Set]
    y = y_train[:Size_of_Data_Set]

    # define model dictionary and number of folds
    model = dict()
    kf = KFold(n_splits=20, random_state=None, shuffle=True)

    # train each model and store in model dictionary
    for train_index, test_index in kf.split(X):
        mof_desc_train, mof_desc_test = X[train_index], X[test_index]
        mof_prop_train, mof_prop_test = y[train_index], y[test_index]

        for name, estimator in ESTIMATORS.items():
            model[name] = estimator.fit(mof_desc_train, mof_prop_train)

    # print('name, test_score,test_r2_score, test_aue,test_rmse,test_kt, test_ev,test_med\n')

    # define statistics dictionaries
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

    # calculate statistics for each model and print
    for name, estimator in ESTIMATORS.items():
        y_test_predict[name] = estimator.predict(X_unseen)
        test_score[name] = model[name].score(X_unseen, y_unseen)
        test_r2_score[name] = r2_score(y_unseen, y_test_predict[name])
        test_aue[name] = mean_absolute_error(y_unseen, y_test_predict[name])
        test_rmse[name] = mean_squared_error(y_unseen, y_test_predict[name])
        rmse[name] = test_rmse[name]**(.5)
        test_kt[name] = stats.kendalltau(y_unseen, y_test_predict[name])
        test_ev[name] = explained_variance_score(y_unseen, y_test_predict[name])
        test_med[name] = median_absolute_error(y_unseen, y_test_predict[name])

        plt.scatter(y_unseen, y_test_predict[name])
        plt.plot([0, 800], [0, 800], color='red')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()

        data = {'Actual': y_unseen, 'Predicted': y_test_predict[name]}
        df = pd.DataFrame(data, columns=['Actual', 'Predicted'])
        df.to_csv(name + '_predictions.csv', index=False)

        print('{}\nScore: {}\nR2 Score: {}\nAbsolute Mean Error: {}\nRMS Error: {}\nKendall Tau: {}\nExplained Variance : {}\nMedian Absolute Error: {}\n'.format(name, test_score[name], test_r2_score[name], test_aue[name], rmse[name], test_kt[name][0], test_ev[name], test_med[name]))


if __name__ == '__main__':
    filename = ''
    if sys.argv[1] is not None:
        filename = sys.argv[1]
    else:
        filename = input('Enter the data\'s relative path: ')
    main(filename)
    