import os
import sys
import pandas as pd
from mofid.run_mofid import cif2mofid
from rdkit.Chem.Descriptors import *

PATHNAME = os.path.abspath('.') + "/"


def get_features(folder):
    featurization_list = []
    for filename in os.listdir(folder + "cifs/"):
        try:
            featurization = {}
            featurization["filename"] = filename
            mofid = cif2mofid(folder + "cifs/" + filename)
            for i in range(len(mofid["smiles_nodes"])):
                featurization["node" + str(i)] = mofid["smiles_nodes"][i]
            for i in range(len(mofid["smiles_linkers"])):
                featurization["linker" + str(i)] = mofid["smiles_linkers"][i]
            print("here")
            featurization["topology"] = mofid["topology"]
            print(mofid["topology"])
            featurization_list.append(featurization)
        except:
            continue
    df = pd.DataFrame(featurization_list)
    df = df.sort_values(by=["filename"])
    df.to_csv(folder + "mofid_featurization.csv", index=False)


if __name__ == "__main__":
    folder = sys.argv[1]
    get_features(PATHNAME + folder + "/")
