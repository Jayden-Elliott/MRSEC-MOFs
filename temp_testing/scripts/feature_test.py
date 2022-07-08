import os
import sys
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

PATHNAME = os.path.abspath(".") + "/"
FILE_DIR = os.path.dirname(__file__)

def get_run(params):
    run, test_split, Size_of_Data_Set, X_orig, y_orig, file = params
    estimators = {
        "random_forest": RandomForestRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "bagging": BaggingRegressor()
    }

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=test_split, random_state=run)
    X = X_train[:Size_of_Data_Set]
    y = y_train[:Size_of_Data_Set]

    # train models
    kf = KFold(n_splits=10, random_state=None, shuffle=True)
    fold_index = 0
    for train_index, test_index in kf.split(X):
        for name, estimator in estimators.items():
            # print(f"\x1b[1K\rTest Split {test_split}, Run {run}, Fold {fold_index}, Training {name}", end="")
            print(f"Test Split {test_split}, Run {run}, Fold {fold_index}, Training {name}")
            estimator.fit(X[train_index], y[train_index])
        fold_index += 1

    # make predictions and calculate statistics
    prediction = {"true": y_test}
    r2 = {"test_split": test_split}
    mae = {"test_split": test_split}
    for name, estimator in estimators.items():
        prediction[name] = estimator.predict(X_test)
        r2[name] = r2_score(y_test, prediction[name])
        mae[name] = mean_absolute_error(y_test, prediction[name])

    # save predictions for each model as a csv
    predictions_df = pd.DataFrame(prediction)
    predictions_df.to_csv(os.path.join(os.path.dirname(__file__), f"predictions/{os.path.basename(file).split('.')[0]}_{str(test_split * 100).split('.')[0]}_{str(run)}.csv"), index=False)
    
    # return r2 and mae
    return (r2, mae)


def get_test_split(params):
    test_split, Size_of_Data_Set, X_orig, y_orig, file = params
    pool = Pool(5)
    stats = list(pool.map(get_run, [(run, test_split, Size_of_Data_Set, X_orig, y_orig, file) for run in range(5)]))
    r2_runs = [pair[0] for pair in stats]
    mae_runs = [pair[1] for pair in stats]

    r2_by_split = {"test_split": test_split}
    for model in r2_runs[0]:
        if model != "test_split":
            r2_by_split[model] = np.mean([run[model] for run in r2_runs])

    mae_by_split = {"test_split": test_split}
    for model in mae_runs[0]:
        if model != "test_split":
            mae_by_split[model] = np.mean([run[model] for run in mae_runs])

    return (r2_by_split, mae_by_split)


def main(file: str, uniform=False):
    feature_list = pd.read_csv(file)

    # convert feature_list to a dataframe and add temps
    df = pd.DataFrame(feature_list)
    temps = ""
    if not uniform:
        temps = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/ml/kulik_temp_list.csv")
        df = pd.merge(temps, df, on="name", how="outer")
    else:
        temps = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/ml/kulik_temp_list_uniform.csv")
        df = pd.merge(temps, df, on="name", how="left")

    # prepare dataframe for ML
    for col in df.columns:
        if col != "name":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)
    for i in range(1, len(df.columns)):
        if df.columns[i] != "name":
            df.iloc[:, i] = df.iloc[:, i].replace(0, df.iloc[:, i].median())
    
    print(len(df.columns))

    names = df.pop("name").tolist()
    df = df.values

    # separate X and y data
    Size_of_Data_Set = len(df)
    X_orig = df[:, 1:]
    y_orig = np.ravel(df[:, 0])

    # scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    # X_orig = scaler.fit_transform(X_orig)
    
    pool = Pool(5)
    stat_avgs = pool.map(get_test_split, [(test_split, Size_of_Data_Set, X_orig, y_orig, file) for test_split in [0.1, 0.15, 0.2, 0.25, 0.3]])
    r2_avgs = [pair[0] for pair in stat_avgs]
    mae_avgs = [pair[1] for pair in stat_avgs]

    # save r2 and mae for all test splits to csvs
    r2_df = pd.DataFrame(r2_avgs)
    r2_df.to_csv(os.path.join(FILE_DIR, f"r2_scores/{os.path.basename(file).split('_features')[0]}.csv"), index=False)
    mae_df = pd.DataFrame(mae_avgs)
    mae_df.to_csv(os.path.join(FILE_DIR, f"mae_scores/{os.path.basename(file).split('_features')[0]}.csv"), index=False)


if __name__ == "__main__":
    file = sys.argv[1]
    uniform = sys.argv[2] if len(sys.argv) > 2 else False
    main(os.path.join(PATHNAME, file), uniform=uniform)
