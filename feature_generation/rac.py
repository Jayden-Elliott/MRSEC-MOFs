import os
import sys
import pandas as pd
from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive, get_MOF_descriptors

PATHNAME = os.path.abspath('.') + '/'

'''
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
'''


def main(folder):
    print("Generating RAC features...")
    if not os.path.exists(folder + "primitives/"):
        os.mkdir(folder + "primitives/")
    if not os.path.exists(folder + "xyz/"):
        os.mkdir(folder + "xyz/")
    
    featurization_list = []
    iter = 0
    for filename in os.listdir(folder + "cifs/"):
        try:
            get_primitive(folder + "cifs/" + filename,
                            folder + "primitives/" + filename)
            # depth = 3
            full_names, full_descriptors = get_MOF_descriptors(folder + "primitives/" + filename, 3, path=folder, xyzpath=folder + "xyz/" + filename.replace("cif", "xyz"))
            full_names.append("name")
            full_descriptors.append(filename.split(".")[0])
            featurization = dict(zip(full_names, full_descriptors))
            featurization_list.append(featurization)
        except:
            continue
        iter += 1
        print(str(iter))
    df = pd.DataFrame(featurization_list)
    names = df.pop("name")
    df.insert(0, "name", names)
    df.to_csv(folder + "rac_features.csv", index=False)
    return df.sort_values(by=["name"])


if __name__ == '__main__':
    folder = sys.argv[1]
    main(PATHNAME + folder + '/')
