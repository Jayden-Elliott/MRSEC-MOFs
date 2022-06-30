import os
import sys
import pandas as pd

database = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/feature_generation/geometric_feature_database.csv")
mofs = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/ml/kulik_temp_list.csv")

output = pd.merge(mofs, database, on="name", how="right")
output.pop("Temp")
output.to_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/data/kulik/geometric_features.csv", index=False)
