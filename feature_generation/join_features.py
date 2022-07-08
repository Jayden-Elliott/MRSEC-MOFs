import os
import sys
import pandas as pd


PATHNAME = os.path.abspath(".") + "/"


def main(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.merge(df1, df2, on="name", how="outer")
    df.to_csv(f"{file1.split('_features')[0]}_and_{os.path.basename(file2)}", index=False)


if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    main(os.path.join(PATHNAME, file1), os.path.join(PATHNAME, file2))