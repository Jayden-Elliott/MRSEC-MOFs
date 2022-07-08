import os
import sys
import pandas as pd

df = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/data/kulik/geometric_features.csv")

for index, row in df.iterrows():
    drop_row = True
    for col in df.columns[1:]:
        if int(row[col]) != 0:
            drop_row = False
            break
    if drop_row:
        df.drop(index, inplace=True)

df.to_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/data/kulik/geometric_no_blanks_features.csv", index=False)
