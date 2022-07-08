import os
import sys
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

feature_and_split = sys.argv[1]

df = pd.read_csv(f"/Users/jayden/Documents/Code/MOFS_MRSEC/temp_testing/predictions/{feature_and_split}_0.csv")

for i in range(1, 5):
    new_df = pd.read_csv(f"/Users/jayden/Documents/Code/MOFS_MRSEC/temp_testing/predictions/{feature_and_split}_{i}.csv")
    df = pd.concat([df, new_df])

df.to_csv(f"/Users/jayden/Documents/Code/MOFS_MRSEC/temp_testing/predictions/{feature_and_split}_all.csv", index=False)
