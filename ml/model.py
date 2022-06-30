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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error
import scipy.stats as spstats

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

PATHNAME = os.path.abspath(".") + "/"

def train_cv(ESTIMATORS, X, y, cv=10):
    # define model dictionary and number of folds
    kf = KFold(n_splits=cv, random_state=None, shuffle=True)

    # train each model and store in model dictionary
    count = 0
    for train_index, test_index in kf.split(X):
        mof_desc_train, mof_desc_test = X[train_index], X[test_index]
        mof_prop_train, mof_prop_test = y[train_index], y[test_index]

        for name, estimator in ESTIMATORS.items():
            estimator.fit(mof_desc_train, mof_prop_train)
            print(name)
        count += 1
        print(f"{count}/10")
    return ESTIMATORS

def train_grid(ESTIMATORS, X, y, param_grid, cv=10):
    grids = {}
    for name, estimator in ESTIMATORS.items():
        grids[name] = GridSearchCV(estimator, param_grid[name], refit=True, cv=cv, verbose=2, n_jobs=-1)
        grids[name].fit(X, y)
        with open(PATHNAME + f"grid_search_{name}.txt", "w") as f:
            f.write("name,best_params,best_score")
            f.write(f"{name},{grids[name].best_params_},{grids[name].best_score_}")
    return grids


# read input csv to pandas dataframe
# csv must have names in col 0, y_true in col 2, and features in the remaining cols
def main(filename):
    mof = pd.read_csv(filename)
    
    for col in mof.columns:
        if mof[col].dtype == "object" and col != "name":
            mof.pop(col)


    """ good_feats_df = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/feature_sets/feature_r2s.csv")
    good_feats = good_feats_df["feature"].tolist()
    
    mof = mof[["name", "Temp"] + good_feats]
    print(len(mof.columns)) """
    
    # names = mof.pop("name")
    # mof.astype("float64")
    # mof.insert(0, "name", names)

    # replace missing values with median
    mof = mof.fillna(0)
    for i in range(1, len(mof.columns)):
        if mof.columns[i] != "name":
            mof.iloc[:, i] = mof.iloc[:, i].replace(0, mof.iloc[:, i].median())

    # convert dataframe to numpy array and remove column names
    names = mof.pop("name").tolist()
    mof = mof.values

    # create X and y arrays
    Size_of_Data_Set = len(mof)
    X_orig = mof[:, 1:]
    y_orig = np.ravel(mof[:, 0])

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_orig = scaler.fit_transform(X_orig)

    rng = np.random.RandomState(100)

    # define ml models to be used
    ESTIMATORS = {
        "random_forest": RandomForestRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "bagging": BaggingRegressor(),
        # "decision_tree": DecisionTreeRegressor(),
        # "linear_regression": LinearRegression(),
        # "ada_boost": AdaBoostRegressor(),
        # "sgd": SGDRegressor(),
        # "ridge": Ridge(),
        # "passive_aggressive": PassiveAggressiveRegressor(),
        # "lasso_lars": LassoLars(),
        # "bayesian_ridge": BayesianRidge(),
        # "ard": ARDRegression(),
        ### "ransac": RANSACRegressor(),
        ### "theil_sen": TheilSenRegressor(),
        ### "lars": Lars(),
        ### "orthogonal_matching_pursuit": OrthogonalMatchingPursuit(),
    }

    # define parameters for grid search
    param_grid = {
        "random_forest": {
            "n_estimators": [2000],
            "max_depth": [10],
            "min_samples_split": [60],
            "max_features": ["sqrt"],
            "bootstrap": [True]
        },
        "gradient_boosting": {
            "n_estimators": [200],
            "max_depth": [10],
            "min_samples_leaf": [50],
            "min_samples_split": [60],
            "max_features": [None],
            "learning_rate": [0.1]
        },
        "bagging": {
            "n_estimators": [1000],
            "max_features": [10],
            "bootstrap": [False]
        }
    }

    # split data into train and test groups
    X_split = X_orig
    y_split = y_orig
    X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.20, random_state=1)
    X_unseen = X_test
    y_unseen = y_test

    N = names[:Size_of_Data_Set]
    X = X_train[:Size_of_Data_Set]
    y = y_train[:Size_of_Data_Set]

    models = train_cv(ESTIMATORS, X, y)
    # models = train_grid(ESTIMATORS, X, y, param_grid)
    
    out_index = 0
    while os.path.exists(PATHNAME + f"output_{out_index}/"):
        out_index += 1
    os.mkdir(PATHNAME + f"output_{out_index}/")

    # calculate statistics for each model and print
    stats_list = []
    stats_list.append({"name": "correct", "y_predict": y_unseen})
    for name, estimator in models.items():
        stats = {"name": name}
        stats["y_predict"] = estimator.predict(X_unseen)
        stats["r2"] = r2_score(y_unseen, stats["y_predict"])
        stats["aue"] = mean_absolute_error(y_unseen, stats["y_predict"])
        stats["mse"] = mean_squared_error(y_unseen, stats["y_predict"])
        stats["rmse"] = stats["mse"]**(.5)
        stats["kt"] = spstats.kendalltau(y_unseen, stats["y_predict"])
        stats["ev"] = explained_variance_score(y_unseen, stats["y_predict"])
        stats["med"] = median_absolute_error(y_unseen, stats["y_predict"])

        plt.scatter(y_unseen, stats["y_predict"])
        plt.plot([0, 1000], [0, 1000], color="red")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)
        plt.savefig(PATHNAME + f"output_{out_index}/{name}.png")

        print(f"{stats.get('name')}\nR2 Score: {stats['r2']}\nAbsolute Mean Error: {stats['aue']}\nMS Error: {stats['mse']}\nRMS Error: {stats['rmse']}\nKendall Tau: {stats['kt']}\nExplained Variance: {stats['ev']}\nMedian Absolute Error: {stats['med']}\n")
        stats_list.append(stats)

    df = pd.DataFrame(stats_list)
    df.to_csv(PATHNAME + f"output_{out_index}/stats.csv")

if __name__ == "__main__":
    filename = ""
    if sys.argv[1] is not None:
        filename = sys.argv[1]
    else:
        filename = input("Enter the data\"s relative path: ")
    main(filename)
    