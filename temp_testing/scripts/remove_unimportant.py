import os
import sys
import pandas as pd
import numpy as np

features = pd.read_csv(os.path.join(os.path.abspath("."), sys.argv[1]))
importances = pd.read_csv(os.path.join(os.path.abspath("."), sys.argv[2]))

for index, row in importances.iterrows():
    if row["importance"] <= 0:
        try:
            features.drop(row["feature"], axis=1, inplace=True)
        except:
            pass

features.to_csv(os.path.join(os.path.abspath("."), f"{sys.argv[1].split('.')[0]}_important.csv"), index=False)