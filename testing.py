import os
import sys
import pandas as pd
from pymatgen.core.structure import IMolecule
from mp_api import MPRester
import re
from pymatgen.core import Element

PATHNAME = os.path.abspath('.') + "/"
MP_API_KEY = "tfZvJ83AZZujeP7q5ZA4VcXOsieXYg4t"


def get_features(filename):
    mol = IMolecule.from_file(filename)
    if mol.composition.contains_element_type("metal") == False:
        return
    featurization = {}
    featurization["formula"] = mol.formula
    featurization["mass"] = mol.composition.weight
    featurization["mass_per_metal"] = featurization["mass"] / \
        int(re.search(r"\d+", featurization["formula"]).group())
    metal_symbol = re.search(r".+?(?=\d)", featurization["formula"]).group()
    metal = Element(metal_symbol)
    featurization["metal_melting_point"] = metal.melting_point
    orbitals = metal.atomic_orbitals
    for o in orbitals:
        featurization["metal_" + o + "_energy"] = orbitals[o]
    print(featurization)


if __name__ == "__main__":
    relative_name = sys.argv[1]
    get_features(PATHNAME + relative_name)
