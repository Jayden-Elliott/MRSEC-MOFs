import os
import sys
import pandas as pd
import numpy as np

folder = "/Users/jayden/Documents/Code/MOFS_MRSEC/data/kulik/"

def node_list(folder):
    metals = ["Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"]

    mofs = {}
    index = 0
    for file in os.listdir(os.path.join(folder, "sbus/")):
        if file.endswith(".xyz"):
            print("node " + str(index))
            index += 1
            metal = ""
            with open(os.path.join(folder, "sbus", file), "r") as f:
                for line in f.readlines():
                    for m in metals:
                        if m in line:
                            metal = m

            if file.split("_sbu")[0] not in mofs.keys():
                mofs[file.split("_sbu")[0]] = [(file, metal)]
                continue
            
            exists = False
            for pair in mofs[file.split("_sbu")[0]]:
                if metal in pair:
                    exists = True
            if not exists:
                mofs[file.split("_sbu")[0]].append((file, metal))
                
    return mofs

def linker_list(folder):
    mofs = {}
    index = 0
    for file in os.listdir(os.path.join(folder, "linkers")):
        if file.endswith(".xyz"):
            print("linker " + str(index))
            index += 1

            elements = {}
            with open(os.path.join(folder, "linkers", file), "r") as f:
                for line in f.readlines():
                    if len(line) <= 3:
                        continue
                    if line.split()[0] not in elements.keys():
                        elements[line.split()[0]] = 1
                    else:
                        elements[line.split()[0]] += 1

            if file.split("_linker")[0] not in mofs.keys():
                mofs[file.split("_linker")[0]] = [(file, elements)]
                continue
            
            exists = False
            for pair in mofs[file.split("_linker")[0]]:
                if elements == pair[1]:
                    exists = True
            if not exists:
                mofs[file.split("_linker")[0]].append((file, elements))

    return mofs

def print_results(part_dict, part):
    with open(os.path.join(os.path.abspath("."), part + "_file_pairs.csv"), "w") as f:
        f.write(f"name,{part}1,{part}2\n")
        for key, val in part_dict.items():
            if len(val) == 1:
                f.write(f"{key},{val[0][0]},{val[0][0]}\n")
            else:
                try:
                    f.write(f"{key},{val[0][0]},{val[1][0]}\n")
                except:
                    print(key)
                    print(val)
                    sys.exit()

if __name__ == "__main__":
    node_dict = node_list(folder)
    linker_dict = linker_list(folder)
    print_results(node_dict, "node")
    print_results(linker_dict, "linker")
