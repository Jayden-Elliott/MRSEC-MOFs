import os
import sys
import pandas as pd

PATHNAME = os.path.abspath(".") + "/"

def main(filename):
    df = pd.read_csv(filename, low_memory=False)
    kulik_raw = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/ml/kulik_temp_list.csv")
    df = df.merge(kulik_raw, on="name", how="outer")

    temps = df.pop("T")
    df.insert(1, "Temp", temps)
    df.to_csv(filename.split(".csv")[0] + "_with_T.csv", index=False)

if __name__ == "__main__":
    filename = sys.argv[1]

    main(os.path.join(PATHNAME, filename))