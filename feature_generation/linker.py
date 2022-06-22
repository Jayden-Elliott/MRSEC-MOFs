import os
import sys
import pandas as pd
from rdkit.Chem import MolFromMolFile
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs.cDataStructs import IntSparseIntVect

PATHNAME = os.path.abspath(".") + "/"


def main(folder):
    print("Generating linker features...")
    featurization_list = []
    lf = folder + "linkers/"

    # gets list of descriptor functions from rdMolDescriptors
    feat_funcs = []
    blocklist = ["AUTOCORR", "CalcMolFormula", "CalcCoulombMat", "Fingerprint", "_CalcLabuteASAContribs", "CalcGETAWAY", "CalcEEMcharges", "GetConnectivityInvariants", "GetFeatureInvariants", "_CalcCrippenContribs", "_CalcTPSAContribs"]
    for i in dir(rdMolDescriptors):
        func = getattr(rdMolDescriptors, i)
        if not callable(func): continue
        allowed = True
        for block in blocklist:
            if block in func.__name__:
                allowed = False
        if allowed:
            feat_funcs.append(func)

    iter = 0
    for filename in sorted(filter(lambda x: os.path.isfile(os.path.join(lf, x)), os.listdir(lf))):
        try:
            if filename.endswith(".xyz"):
                # converts xyz file to mol file using openbabel if the mol doesn"t exist
                if not os.path.exists(lf + filename.replace(".xyz", ".mol")):
                    os.system("obabel -ixyz \"" + lf + filename + "\" -omol -O \"" + lf + filename.replace(".xyz", ".mol") + "\"")

                featurization = {}
                featurization["name"] = filename[:filename.index("_linker")]
                try:
                    # creates mol object from mol file produced by openbabel
                    linker = MolFromMolFile(lf + filename.replace(".xyz", ".mol"))

                    # calculates features for mol object
                    for func in feat_funcs:
                        try:
                            output = func(linker)
                            if isinstance(output, list) or isinstance(output, tuple):
                                for i in range(len(output)):
                                    if isinstance(output[i], tuple):
                                        for j in range(len(output[i])):
                                            featurization[func.__name__ + "-" + str(i) + "-" + str(j)] = output[i][j]
                                    else:
                                        featurization[func.__name__ + "-" + str(i)] = output[i]
                            else:
                                featurization[func.__name__] = output
                        except:
                            continue

                except:
                    featurization_list.append(featurization)
                    continue

                # checks if the linker is a duplicate of one already featurized
                if featurization["name"].endswith("-primitive"):
                    for f in featurization_list:
                        if f["name"] == featurization["name"][:featurization["name"].index("-primitive")] and round(f["_CalcMolWt"], 1) == round(featurization["_CalcMolWt"], 1):
                            featurization_list.remove(f)
                            featurization["name"] = featurization["name"][:featurization["name"].index("-primitive")]
                            break

                duplicate = False
                for f in featurization_list:
                    if round(f["_CalcMolWt"], 1) == round(featurization["_CalcMolWt"], 1):
                        if f["name"] == featurization["name"]:
                            duplicate = True
                        if f["name"].endswith("-primitive") and f["name"][:f["name"].index("-primitive")] == featurization["name"]:
                            duplicate = True
                            f["name"] = featurization["name"]

                if not duplicate:
                    featurization_list.append(featurization)
        except:
            continue
        iter += 1
        if iter % 100 == 0:
            print(str(iter))
    
    #combines mofs with multiple linkers
    feats_by_name = {}
    for f in featurization_list:
        if f["name"] not in feats_by_name:
            feats_by_name[f["name"]] = [f]
        else:
            feats_by_name[f["name"]].append(f)
    
    expanded_featurization_list = []

    for name in feats_by_name:
        expanded_featurization_list.append({"name": name})

        # mofs with only one linker have it duplicated
        if len(feats_by_name[name]) == 1:
            for feat in feats_by_name[name][0]:
                if feat == "name":
                    continue
                expanded_featurization_list[-1][feat + "_0"] = feats_by_name[name][0][feat]
                expanded_featurization_list[-1][feat + "_1"] = feats_by_name[name][0][feat]
        
        # mofs with multiple linkers have the linkers averaged into two feature lists
        else:
            max_index = 0
            min_index = 0
            for feat_list in feats_by_name[name]:
                if feat_list["_CalcMolWt"] > feats_by_name[name][max_index]["_CalcMolWt"]:
                    max_index = feats_by_name[name].index(feat_list)
                if feat_list["_CalcMolWt"] < feats_by_name[name][min_index]["_CalcMolWt"]:
                    min_index = feats_by_name[name].index(feat_list)
            
            #create two feature lists for the linkers with the most similar weights
            max_list = []
            min_list = []
            for feat_list in feats_by_name[name]:
                if abs(feat_list["_CalcMolWt"] - feats_by_name[name][max_index]["_CalcMolWt"]) < abs(feat_list["_CalcMolWt"] - feats_by_name[name][min_index]["_CalcMolWt"]):
                    max_list.append(feat_list)
                else:
                    min_list.append(feat_list)
            
            #average the features of the two feature lists
            for feat in feats_by_name[name][0]:
                if feat == "name":
                    continue
                misses = 0
                max_avg = 0
                for feat_list in max_list:
                    if feat in feat_list:
                        max_avg += feat_list[feat]
                    else:
                        misses += 1
            
                if len(max_list) - misses == 0:
                    max_avg = max_list[0].get(feat) if len(max_list) > 0 else 0
                else:
                    max_avg /= (len(max_list) - misses)

                misses = 0
                min_avg = 0
                for feat_list in min_list:
                    if feat in feat_list:
                        min_avg += feat_list[feat]
                    else:
                        misses += 1

                if len(min_list) - misses == 0:
                    min_avg = min_list[0].get(feat) if len(min_list) > 0 else 0
                else:
                    min_avg /= (len(min_list) - misses)

                expanded_featurization_list[-1][feat + "_0"] = max_avg
                expanded_featurization_list[-1][feat + "_1"] = min_avg

    df = pd.DataFrame(expanded_featurization_list)
    df.to_csv(folder + "linker_features.csv", index=False)
    return df.sort_values(by=["name"])


if __name__ == "__main__":
    folder = sys.argv[1]
    main(PATHNAME + folder + "/")
