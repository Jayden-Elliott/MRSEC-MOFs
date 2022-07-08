import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import linregress

PATHNAME = os.path.abspath(".") + "/"

def main(filename):
    df = pd.read_csv(filename, low_memory=False)

    for col in df.columns:
        if df[col].dtype == "object" and col != "name":
            df.pop(col)

    df = df.fillna(0)
    for i in range(1, len(df.columns)):
        df.iloc[:, i] = df.iloc[:, i].replace(0, df.iloc[:, i].median())

    r2s = [(feature, linregress(df[feature].tolist(), df["Temp"].tolist())[2]**2) for feature in df.columns if (feature != "Temp" and feature != "name")]
    r2s_df = pd.DataFrame(r2s, columns=["feature", "r2"])
    r2s_df.sort_values(by="r2", ascending=False, inplace=True)
    r2s_df.to_csv("feature_r2s.csv", index=False)


if __name__ == "__main__":
    filename = sys.argv[1]
    main(os.path.join(PATHNAME, filename))