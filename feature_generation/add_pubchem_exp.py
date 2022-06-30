import pandas as pd

df = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/data/kulikfeatures/all_features_with_T_1.csv", low_memory=False)
exp = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/data/kulik/pubchem_exp_features.csv")
df = df.merge(exp, on="name")

df.to_csv("all_features_with_T.csv", index=False)