import os
import sys
import pandas as pd
import re
from pymatgen.core.structure import IMolecule
from pymatgen.core import Element

PATHNAME = os.path.abspath('.') + '/'


def main(folder):
    print("Generating node features...")
    sbus = folder + 'sbus/'
    if not os.path.exists(sbus):
        return

    element_features = pd.read_csv(os.path.join(os.path.dirname(__file__), 'element_features.csv'))

    featurization_list = []
    iter = 0
    for filename in os.listdir(sbus):
        try:
            if filename.endswith('.xyz'):
                featurization = {}
                mol = IMolecule.from_file(sbus + filename)
                if mol.composition.contains_element_type('metal') == False:
                    break

                featurization['name'] = filename.split('_sbu_')[0]
                featurization['formula'] = mol.formula
                featurization['mass'] = mol.composition.weight
                featurization['mass_per_metal'] = featurization['mass'] / int(re.search(r'\d+', featurization['formula']).group())
                metal_symbol = re.search(
                    r'.+?(?=\d)', featurization['formula']).group()
                metal = Element(metal_symbol)
                featurization['metal_melting_point'] = metal.melting_point
                orbitals = metal.atomic_orbitals
                for o in orbitals:
                    featurization['metal_' + o + '_energy'] = orbitals[o]

                for feat in element_features.columns:
                    featurization[feat] = element_features.loc[element_features["Sym"] == metal_symbol, feat].values[0]
                
                symbols = {"oxygen": "O", "nitrogen": "N", "carbon": "C"}

                for element in symbols:
                    try:
                        element_count = 0
                        if symbols[element] in featurization["formula"]:
                            element_count = int(re.search(symbols[element] + r"(\d+)", featurization["formula"]).group()[1:])
                        featurization[element + "_count"] = element_count
                    except:
                        continue

                # checks if the linker is a duplicate of one already featurized
                if featurization["name"].endswith("-primitive"):
                    for f in featurization_list:
                        if f["name"] == featurization["name"][:featurization["name"].index("-primitive")]:
                            featurization_list.remove(f)
                            featurization["name"] = featurization["name"][:featurization["name"].index("-primitive")]
                            break
                
                duplicate = False
                for f in featurization_list:
                    if f["name"] == featurization["name"]:
                        duplicate = True
                    if f["name"].endswith("-primitive") and f["name"][:f["name"].index("-primitive")] == featurization["name"]:
                        duplicate = True
                        f["name"] = featurization["name"]

                if not duplicate:
                    featurization_list.append(featurization)
        except Exception as e:
            print(e)
            continue
        iter += 1
        print(str(iter))

    df = pd.DataFrame(featurization_list)
    df.to_csv(folder + "node_features.csv", index=False)
    return df.sort_values(by=['name'])


if __name__ == '__main__':
    folder = sys.argv[1]
    main(PATHNAME + folder + '/')
