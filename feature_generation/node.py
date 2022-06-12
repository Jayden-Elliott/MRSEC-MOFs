import os
import sys
import pandas as pd
import re
from pymatgen.core.structure import IMolecule
from pymatgen.core import Element

PATHNAME = os.path.abspath('.') + '/'


def main(folder):
    sbus = folder + 'sbus/'
    if not os.path.exists(sbus):
        return

    featurization_list = []
    for filename in os.listdir(folder + 'sbus/'):
        try:
            if filename.endswith('.xyz'):
                featurization = {}
                mol = IMolecule.from_file(sbus + filename)
                if mol.composition.contains_element_type('metal') == False:
                    break

                featurization['name'] = filename.split('_sbu_')[0]
                featurization['formula'] = mol.formula
                featurization['mass'] = mol.composition.weight
                featurization['mass_per_metal'] = featurization['mass'] / \
                    int(re.search(r'\d+', featurization['formula']).group())
                metal_symbol = re.search(
                    r'.+?(?=\d)', featurization['formula']).group()
                metal = Element(metal_symbol)
                featurization['metal_melting_point'] = metal.melting_point
                orbitals = metal.atomic_orbitals
                for o in orbitals:
                    featurization['metal_' + o + '_energy'] = orbitals[o]

                # checks if the linker is a duplicate of one already featurized
                duplicate = False
                for f in featurization_list:
                    if f['name'] != featurization['name']:
                        continue
                    all_match = True
                    for key in featurization:
                        if (key != 'name' or 'formula') and round(f[key], 2) != round(featurization[key], 2):
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
    df = df.sort_values(by=['name'])
    df.to_csv(folder + 'metal_featurization.csv', index=False)


if __name__ == '__main__':
    folder = sys.argv[1]
    main(PATHNAME + folder + '/')
