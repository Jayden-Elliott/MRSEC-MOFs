import os
import sys
import pandas as pd

PATHNAME = os.path.abspath(".") + "/"

def main(folder):
    df = pd.read_csv(folder + "all_features.csv", low_memory=False)
    kulik_raw = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/ml/full_TSD_data.csv")
    kulik_raw = kulik_raw.filter(["CoRE_name", "T"])
    kulik_raw.rename(columns={"CoRE_name": "name"}, inplace=True)
    df = df.merge(kulik_raw, on="name")

    

    temps = df.pop("T")
    df.insert(1, "Temp", temps)
    df.to_csv(folder + "all_features_with_T.csv", index=False)

if __name__ == "__main__":
    folder = sys.argv[1]

    main(PATHNAME + folder + "/")