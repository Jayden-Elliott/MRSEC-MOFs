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
import re
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error
import scipy.stats as spstats

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

PATHNAME = os.path.abspath(".") + "/"

# read input csv to pandas dataframe
# csv must have names in col 0, y_true in col 2, and features in the remaining cols
def main(filename):
    mof = pd.read_csv(filename)
    
    for col in mof.columns:
        if mof[col].dtype == "object":
            mof.pop(col)
    
    # names = mof.pop("name")
    # mof.astype("float64")
    # mof.insert(0, "name", names)

    # replace missing values with median
    mof = mof.fillna(0)
    for i in range(1, len(mof.columns)):
        mof.iloc[:, i] = mof.iloc[:, i].replace(0, mof.iloc[:, i].median())

    # convert dataframe to numpy array and remove column names
    names = list(mof.columns[0:])
    mof = mof.values
    name = mof[:, 0]

    # create X and y arrays
    Size_of_Data_Set = len(mof)
    meta_data_1 = mof[:, 0:2]
    X_orig = mof[:, 2:]
    y_orig = np.ravel(mof[:, 2])

    scaler = MinMaxScaler(feature_range=(0, 1))
    # X_orig = scaler.fit_transform(X_orig)

    rng = np.random.RandomState(100)

    # define ml models to be used
    ESTIMATORS = {
        "decision_tree": DecisionTreeRegressor(),
        "linear_regression": LinearRegression(),
        'random_forest': RandomForestRegressor(),
        'gradient_boosting': GradientBoostingRegressor(),
        'ada_boost': AdaBoostRegressor(),
        'bagging': BaggingRegressor(),
        'sgd': SGDRegressor(),
        'ridge': Ridge(),
        'passive_aggressive': PassiveAggressiveRegressor(),
        # 'ransac': RANSACRegressor(),
        # 'theil_sen': TheilSenRegressor(),
        # 'lars': Lars(),
        'lasso_lars': LassoLars(),
        # 'orthogonal_matching_pursuit': OrthogonalMatchingPursuit(),
        'bayesian_ridge': BayesianRidge(),
        'ard': ARDRegression(),
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
    kf = KFold(n_splits=10, random_state=None, shuffle=True)

    # train each model and store in model dictionary
    count = 0
    for train_index, test_index in kf.split(X):
        mof_desc_train, mof_desc_test = X[train_index], X[test_index]
        mof_prop_train, mof_prop_test = y[train_index], y[test_index]

        for name, estimator in ESTIMATORS.items():
            model[name] = estimator.fit(mof_desc_train, mof_prop_train)
            print(name)
        count += 1
        print(f"{count}/10")
        

    # calculate statistics for each model and print
    os.mkdir(PATHNAME + "output/")
    stats_list = []
    stats_list.append({"model": "correct", "data": y_unseen})
    for name, estimator in ESTIMATORS.items():
        stats = {"name": name}
        stats["y_test_predict"] = estimator.predict(X_unseen)
        stats["test_score"] = model[name].score(X_unseen, y_unseen)
        stats["test_r2_score"] = r2_score(y_unseen, stats["y_test_predict"])
        stats["test_aue"] = mean_absolute_error(y_unseen, stats["y_test_predict"])
        stats["test_rmse"] = mean_squared_error(y_unseen, stats["y_test_predict"])
        stats["rmse"] = stats["test_rmse"]**(.5)
        stats["test_kt"] = spstats.kendalltau(y_unseen, stats["y_test_predict"])
        stats["test_ev"] = explained_variance_score(y_unseen, stats["y_test_predict"])
        stats["test_med"] = median_absolute_error(y_unseen, stats["y_test_predict"])

        plt.scatter(y_unseen, stats["y_test_predict"])
        plt.plot([0, 2000], [0, 2000], color="red")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.xlim(0, 2000)
        plt.ylim(0, 2000)
        plt.savefig(PATHNAME + f"output/{name}.png")
        plt.show()

        print("{}\nScore: {}\nR2 Score: {}\nAbsolute Mean Error: {}\nRMS Error: {}\nKendall Tau: {}\nExplained Variance : {}\nMedian Absolute Error: {}\n".format(name, stats["test_score"], stats["test_r2_score"], stats["test_aue"], stats["rmse"], stats["test_kt"][0], stats["test_ev"], stats["test_med"]))
        stats_list.append({"model": name, "data": stats["y_test_predict"]})

    df = pd.DataFrame(stats_list)
    df.to_csv(PATHNAME + "output/stats.csv")

if __name__ == "__main__":
    filename = ""
    if sys.argv[1] is not None:
        filename = sys.argv[1]
    else:
        filename = input("Enter the data\"s relative path: ")
    main(filename)
    