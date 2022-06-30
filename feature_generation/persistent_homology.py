import os
import sys
import pandas as pd
import numpy as np
sys.path.append("/Users/jayden/Documents/Code/MOFS_MRSEC/modules/PersistentImages_Chemistry/")
from Element_PI import VariancePersist
from Element_PI import VariancePersistv1

PATHNAME = os.path.abspath(".") + "/"


def main(folder, size=25, output_path=""):
    if output_path == "":
        output_path = os.path.join(folder, "homology_features.csv")
    print("Generating perisistent homology features...")

    file_list = pd.merge(pd.read_csv(os.path.join(folder, "node_file_pairs.csv")), pd.read_csv(os.path.join(folder, "linker_file_pairs.csv")), on="name", how="outer")

    pixelsx = size
    pixelsy = size
    spread = 0.08
    max = 2.5
    samples = len(file_list)

    feature_list = []
    for mof_index, row in file_list.iterrows():
        if mof_index % 250 == 0:
            print(mof_index)
        mof_index += 1
        name = row["name"]

        features = {"name": name}

        def list_feats(part):
            index = 0

            path = ""
            if part.startswith("node"):
                path = os.path.join(folder, "sbus")
            else:
                path = os.path.join(folder, "linkers")
            
            try:
                for feat in VariancePersistv1(os.path.join(path, row[part]), pixelx=pixelsx, pixely=pixelsy, myspread=spread, myspecs={"maxBD": max, "minBD": -.10}, showplot=False):
                    features[part + "_homology_" + str(index)] = feat
                    index += 1
            except:
                print(name + " failed")

        list_feats("node1")
        list_feats("node2")
        list_feats("linker1")
        list_feats("linker2")

        feature_list.append(features)

    df = pd.DataFrame(feature_list)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    folder = sys.argv[1]

    size = 25
    output_path = os.path.join(PATHNAME, folder, "homology_features.csv")
    if len(sys.argv) > 2:
        size = int(sys.argv[2])
    if len(sys.argv > 3):
        output_path = os.path.join(PATHNAME, sys.argv[3])
    df = main(os.path.join(PATHNAME, folder), size, output_path)
