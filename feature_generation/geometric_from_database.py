import os
import sys
import pandas as pd

database = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/feature_generation/geometric_feature_database.csv")
mofs = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/ml/kulik_temp_list.csv")
mofs.pop("Temp")

feature_lists = {}
for i in range(len(database.columns)):
    if database.columns[i] != "name":
        feature_lists[database.columns[i]] = []
fails = 0
for name in mofs["name"].to_list():
    if name not in database["name"].to_list():
        for col in database.columns[1:]:
            feature_lists[col].append(0)
        fails += 1
    else:
        for col in database.columns[1:]:
            feature_lists[col].append(database[col][database["name"] == name].values[0])

for feature, values in feature_lists.items():
    mofs[feature] = values

print(fails)
mofs.to_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/data/kulik/geometric_features.csv", index=False)
