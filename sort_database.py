import os
import sys
import pandas as pd

df = pd.read_csv("ml/full_TSD_data.csv")
names = df["CoRE_name"].tolist()

for name in names:
    os.system("cp core_database/{}.cif data/kulik/cifs/".format(name))