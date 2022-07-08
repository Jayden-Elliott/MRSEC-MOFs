import os
import sys
import pandas as pd

folder = "/Users/jayden/Documents/Code/MOFS_MRSEC/data/kulik"

sources = ["geometric_features_important.csv", "homology_features_important.csv", "linker_features_important.csv", "node_features_important.csv", "rac_features_important.csv"]

df = pd.read_csv(os.path.join(folder, sources[0]))

for source in sources[1:]:
    df = pd.merge(df, pd.read_csv(os.path.join(folder, source)), on="name", how="outer")

df.to_csv(os.path.join(folder, "all_features_important.csv"), index=False)