import os
import sys
import pandas as pd
from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive, get_MOF_descriptors

PATHNAME = os.path.abspath('.') + "/"

"""
f = full SBU RAC
mc = metal-centered SBU RAC
D_mc = metal-centered SBU delta metric
f-lig = full linker RAC
lc = linker-connecting atoms RAC
D_lc = linker-connecting atoms delta metric
func = functional group RAC
D_func = functional group delta metric

electronegativity = chi
nuclear charge = Z
identity = I
topology = T
size = S

index = depth
"""


def get_features(folder):
    featurization_list = []
    for filename in os.listdir(folder + "cifs/"):
        try:
            get_primitive(folder + "cifs/" + filename,
                          folder + "primitives/" + filename)
            # depth = 3
            full_names, full_descriptors = get_MOF_descriptors(
                folder + "primitives/" + filename, 3, path=folder, xyzpath=folder + "xyz/" + filename.replace("cif", "xyz"))
            full_names.append("filename")
            full_descriptors.append(filename)
            featurization = dict(zip(full_names, full_descriptors))
            featurization_list.append(featurization)
        except:
            continue
    df = pd.DataFrame(featurization_list)
    df = df.sort_values(by=["filename"])
    df.to_csv(folder + "rac_featurization.csv", index=False)


if __name__ == "__main__":
    folder = sys.argv[1]
    get_features(PATHNAME + folder + "/")
