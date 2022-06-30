import os
import sys
import pandas as pd


def get_metals(formula: str):
    metals = ["Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"]
    
    metal1 = ""
    metal2 = ""
    for m in metals:
        if m in formula:
            if metal1 == "":
                metal1 = m
            else:
                metal2 = m
    if metal2 == "":
        metal2 = metal1
    
    return metal1, metal2

node_features = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/data/kulik/node_features.csv")
element_features = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/feature_generation/element_features.csv")

metal1_list = []
metal2_list = []
for index, row in node_features.iterrows():
    metal1, metal2 = get_metals(row["formula"])
    metal1_list.append(metal1)
    metal2_list.append(metal2)

node_features["symbol_1"] = metal1_list
node_features["symbol_2"] = metal2_list
node_features = pd.merge(node_features, element_features, left_on="symbol_1", right_on="symbol", how="outer", suffixes=("_1", "_2"))
node_features = pd.merge(node_features, element_features, left_on="symbol_2", right_on="symbol", how="outer", suffixes=("_1", "_2"))

node_features.to_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/data/kulik/node_elements_features.csv", index=False)
    