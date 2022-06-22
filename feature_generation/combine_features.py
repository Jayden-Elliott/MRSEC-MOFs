import os
import sys
import pandas as pd

PATHNAME = os.path.abspath(".") + "/"

folder = sys.argv[1]
if os.path.exists(folder + "features/"):
    os.system("rm -rf " + folder + "features/")
os.mkdir(folder + "features/")

folder = PATHNAME + folder + "/"
df = pd.read_csv(folder + "rac_features.csv")
df = pd.merge(df, pd.read_csv(folder + "mofid_pubchem_features.csv"), on="name", how="outer", suffixes=("", "_mofidpubchem"))
df = pd.merge(df, pd.read_csv(folder + "linker_features.csv"), on="name", how="outer", suffixes=("", "_linker"))
df = pd.merge(df, pd.read_csv(folder + "node_features.csv"), on="name", how="outer", suffixes=("", "_node"))

df.to_csv(folder + "all_features.csv", index=False)