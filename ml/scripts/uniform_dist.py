import os
import sys
import pandas as pd
import random
import numpy as np

from sqlalchemy import column

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "kulik_temp_list.csv"))
df.sort_values(by="Temp", inplace=True)
df = df[df["Temp"] <= 800]
df["binned"] = pd.cut(df["Temp"], np.linspace(0, 800, 81), labels=False)
bins = {}
for index, row in df.iterrows():
    if row["binned"] not in bins:
        bins[row["binned"]] = []
    bins[row["binned"]].append(index)

for row_list in bins.values():
    while(len(row_list) > 50):
        dropped = random.choice(row_list)
        row_list.pop(row_list.index(dropped))
        df.drop(dropped, inplace=True)

df.drop(columns=["binned"], inplace=True)
df.to_csv(os.path.join(os.path.dirname(__file__), "kulik_temp_list_uniform.csv"), index=False)


