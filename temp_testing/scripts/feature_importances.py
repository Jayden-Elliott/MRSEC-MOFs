import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

PATHNAME = os.path.abspath(".") + "/"


def main(file: str, output: str):
    feature_list = pd.read_csv(file)

    # convert feature_list to a dataframe and add temps
    df = pd.DataFrame(feature_list)
    temps = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/ml/kulik_temp_list.csv")
    df = pd.merge(temps, df, on="name", how="outer")

    # prepare dataframe for ML
    for col in df.columns:
        if col != "name":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)
    for i in range(1, len(df.columns)):
        if df.columns[i] != "name":
            df.iloc[:, i] = df.iloc[:, i].replace(0, df.iloc[:, i].median())
    
    names = df.pop("name").tolist()
    feature_list = df.columns.tolist().remove("Temp")
    df = df.values

    # separate X and y data
    Size_of_Data_Set = len(df)
    X_orig = df[:, 1:]
    y_orig = np.ravel(df[:, 0])

    # scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_orig = scaler.fit_transform(X_orig)

    random_forest = RandomForestRegressor()

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.25)
    X = X_train[:Size_of_Data_Set]
    y = y_train[:Size_of_Data_Set]

    # train model
    kf = KFold(n_splits=10, shuffle=True)
    fold_index = 0
    for train_index, test_index in kf.split(X):
        random_forest.fit(X[train_index], y[train_index])
        print("Fold: " + str(fold_index))
        fold_index += 1
    
    result = permutation_importance(random_forest, X_test, y_test, n_repeats=10, n_jobs=-1, scoring="r2")
    forest_importances = pd.Series(result.importances_mean, index=feature_list)
    forest_importances.sort_values(ascending=False, inplace=True)
    forest_importances.to_csv(f"{output}.csv")

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    ax.set_title(f"{os.path.join(os.path.basename(file).split('.')[0])} Feature Importances")
    fig.tight_layout()
    plt.savefig(f"{output}.png")
    plt.show()

if __name__ == "__main__":
    file = os.path.join(PATHNAME, sys.argv[1])
    output = sys.argv[2]
    main(file, output)