import os
import sys
import pandas as pd
from rdkit.Chem import MolFromMolFile
from rdkit.Chem.Descriptors import ExactMolWt, MaxPartialCharge
from rdkit.Chem.Crippen import MolLogP

PATHNAME = os.path.abspath('.') + "/"


def get_features(folder):
    featurization_list = []
    lf = folder + "linkers/"
    for filename in sorted(filter(lambda x: os.path.isfile(os.path.join(lf, x)), os.listdir(lf))):
        try:
            if filename.endswith(".xyz"):
                # converts xyz file to mol file using openbabel if the mol doesn't exist
                if not os.path.exists(lf + filename.replace(".xyz", ".mol")):
                    os.system("obabel -ixyz \"" + lf + filename +
                              "\" -omol -O \"" + lf + filename.replace(".xyz", ".mol") + "\"")

                featurization = {}
                featurization["name"] = filename[:filename.index("_linker")]
                try:
                    linker = MolFromMolFile(
                        lf + filename.replace(".xyz", ".mol"))
                    featurization["linker_weight"] = ExactMolWt(linker)
                    featurization["linker_max_partial_charge"] = MaxPartialCharge(
                        linker)
                    featurization["linker_logP"] = MolLogP(linker)
                except:
                    featurization_list.append(featurization)
                    continue

                # checks if the linker is a duplicate of one already featurized
                duplicate = False
                for f in featurization_list:
                    if f["name"] != featurization["name"]:
                        continue
                    all_match = True
                    for key in featurization:
                        if key != "name" and round(f[key], 2) != round(featurization[key], 2):
                            all_match = False
                            break
                    if all_match:
                        duplicate = True
                        break

                if not duplicate:
                    featurization_list.append(featurization)
        except:
            continue

    df = pd.DataFrame(featurization_list)
    df.to_csv(folder + "rdkit_featurization.csv", index=False)


if __name__ == "__main__":
    folder = sys.argv[1]
    get_features(PATHNAME + folder + "/")
