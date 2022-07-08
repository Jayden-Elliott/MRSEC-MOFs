import os
import sys
import pandas as pd
import numpy as np

sys.path.append("/Users/jayden/Documents/Code/MOFS_MRSEC/modules/PersistentImages_Chemistry/")
from Element_PI import VariancePersist
from Element_PI import VariancePersistv1

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

PATHNAME = os.path.abspath(".") + "/"


def main(folder, size):
    # merge lists of node files and edge files into one dataframe sorted by mof name
    file_list = pd.merge(pd.read_csv(os.path.join(folder, "node_file_pairs.csv")), pd.read_csv(os.path.join(folder, "linker_file_pairs.csv")), on="name", how="outer")

    # set persistent homology parameters
    pixelsx = int(size)
    pixelsy = int(size)
    spread = 0.08
    max = 2.5
    samples = len(file_list)

    # generate persistent homology features for each mof and add to feature_list
    mof_index = 0
    feature_list = []
    for mof_index, row in file_list.iterrows():
        if mof_index % 250 == 0:
            print(mof_index)
        mof_index += 1

        name = row["name"]
        features = {"name": name}

        # generate features given a mof part
        def list_feats(part):
            index = 0

            # set the path of the xyz file for the mof part
            path = ""
            if part.startswith("node"):
                path = os.path.join(folder, "sbus")
            else:
                path = os.path.join(folder, "linkers")
            
            # generate and add features to the features dictionary
            try:
                for feat in VariancePersistv1(os.path.join(path, row[part]), pixelx=pixelsx, pixely=pixelsy, myspread=spread, myspecs={"maxBD": max, "minBD": -.10}, showplot=False):
                    features[part + "_homology_" + str(index)] = feat
                    index += 1
            except:
                pass
        
        # generate features for the node and linker parts of the mof
        list_feats("node1")
        list_feats("node2")
        list_feats("linker1")
        list_feats("linker2")

        feature_list.append(features)

    # convert feature_list to a dataframe, add temps, and write to csv
    df = pd.DataFrame(feature_list)
    temps = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/ml/kulik_temp_list.csv")
    df = pd.merge(temps, df, on="name", how="outer")
    df.to_csv(f"./feature_sets/{size}.csv", index=False)

    # prepare dataframe for ML
    df = df.fillna(0)
    names = df.pop("name").tolist()
    df = df.values

    # separate X and y data
    Size_of_Data_Set = len(df)
    X_orig = df[:, 1:]
    y_orig = np.ravel(df[:, 0])

    # scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_orig = scaler.fit_transform(X_orig)

    r2_avgs = []
    mae_avgs = []
    for test_split in [0.1, 0.15, 0.2, 0.25, 0.3]:
        print(f"Test split: {test_split}")

        r2_runs = []
        mae_runs = []
        for run in range(3):
            print(f"Run: {run}")

            estimators = {
                "random_forest": RandomForestRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
                "bagging": BaggingRegressor()
            }

            # split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=test_split, random_state=1)
            N = names[:Size_of_Data_Set]
            X = X_train[:Size_of_Data_Set]
            y = y_train[:Size_of_Data_Set]

            # train models
            kf = KFold(n_splits=10, random_state=None, shuffle=True)
            fold_index = 1
            for train_index, test_index in kf.split(X):
                print(f"Training on fold {fold_index}/10")
                fold_index += 1

                for name, estimator in estimators.items():
                    print(f"Training {name}")

                    estimator.fit(X[train_index], y[train_index])

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
            predictions_df.to_csv(os.path.join(os.path.dirname(__file__), f"predictions/{str(size)}_{str(test_split * 100).split('.')[0]}_{str(run)}.csv"), index=False)
            
            # add r2 and mae for this test split to the r2 and mae lists
            r2_runs.append(r2)
            mae_runs.append(mae)
        
        r2_by_split = {"test_split": test_split}
        for model in r2_runs[0]:
            if model != test_split:
                r2_by_split[model] = np.mean([run[model] for run in r2_runs])
        r2_avgs.append(r2_by_split)

        mae_by_split = {"test_split": test_split}
        for model in mae_runs[0]:
            if model != "test_split":
                mae_by_split[model] = np.mean([run[model] for run in mae_runs])
        mae_avgs.append(mae_by_split)
    
    # save r2 and mae for all test splits to csvs
    r2_df = pd.DataFrame(r2_avgs)
    r2_df.to_csv(os.path.join(os.path.dirname(__file__), f"r2_scores/{size}.csv"), index=False)
    mae_df = pd.DataFrame(mae_avgs)
    mae_df.to_csv(os.path.join(os.path.dirname(__file__), f"mae_scores/{size}.csv"), index=False)


if __name__ == "__main__":
    folder = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else 10
    main(os.path.join(PATHNAME, folder), size)
