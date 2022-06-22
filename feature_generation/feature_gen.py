import os
import sys
import pandas as pd

from rac import main as rac_feats
from geometric import main as geometric_feats
from linker import main as linker_feats
from node import main as node_feats
from mofidpubchem import main as mofidpubchem_feats

PATHNAME = os.path.abspath(".") + "/"


if __name__ == "__main__":
    folder = sys.argv[1]
    if os.path.exists(folder + "features/"):
        os.system("rm -rf " + folder + "features/")
    os.mkdir(folder + "features/")

    folder = PATHNAME + folder + "/"
    df = rac_feats(folder)
    df = pd.merge(df, geometric_feats(folder), on="name", how="outer", suffixes=("_rac", "_geo"))
    df = pd.merge(df, mofidpubchem_feats(folder), on="name", how="outer", suffixes=("", "_mofidpubchem"))
    df = pd.merge(df, linker_feats(folder), on="name", how="outer", suffixes=("", "_linker"))
    df = pd.merge(df, node_feats(folder), on="name", how="outer", suffixes=("", "_node"))

    df.to_csv(folder + "all_features.csv", index=False)
