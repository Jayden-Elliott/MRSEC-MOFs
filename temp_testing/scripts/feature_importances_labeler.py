import os
import sys
import pandas as pd
import numpy as np

importances = pd.read_csv(os.path.join(os.path.abspath("."), sys.argv[1]))
features = pd.read_csv(os.path.join(os.path.abspath("."), sys.argv[2]))

features.pop("name")
features = features.columns.tolist()

feature_names = [features[index] for index in importances["feature_index"].tolist()]

importances.insert(1, "feature", feature_names)

importances.to_csv(os.path.join(os.path.abspath("."), sys.argv[1]), index=False)