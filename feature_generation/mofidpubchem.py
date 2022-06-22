import os
import sys
import pandas as pd
from mofid.run_mofid import cif2mofid
from rdkit.Chem.Descriptors import *
import urllib.parse
import requests

PATHNAME = os.path.abspath('.') + '/'


def main(folder):
    print("Generating MOFid and PubChem features...")
    featurization_list = []
    iter = 0
    for filename in os.listdir(folder + "cifs/"):
        try:
            featurization = {}
            featurization["name"] = filename.split(".")[0]
            mofid = cif2mofid(folder + "cifs/" + filename,
                            output_path=folder + "mofid_output/")
            featurization["topology"] = mofid["topology"]
            if len(mofid["smiles_linkers"]) == 0:
                featurization_list.append(featurization)
                continue
            smiles = mofid["smiles_linkers"][0].replace("O-", "OH")
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{urllib.parse.quote(smiles)}/record/JSON"
            response = requests.get(url).json()
            
            blocklist = ["Canonicalized", "IUPAC Name", "InChI", "Molecular Formula"]
            props = response["PC_Compounds"][0]["props"]
            for prop in props:
                skip = False
                for block in blocklist:
                    if block in prop["urn"]["label"]:
                        skip = True
                if skip: continue
                try:
                    feat_name = (((prop["urn"]["name"] + "-") if "name" in prop["urn"] else "") + prop["urn"]["label"]).replace(" ", "-")
                    if "Fingerprint" in feat_name:
                        for i in range(len(prop["value"][list(prop["value"].keys())[0]])):
                            featurization[feat_name + "-" + str(i)] = prop["value"][list(prop["value"].keys())[0]][i]
                    else:
                        featurization[feat_name] = prop["value"][list(prop["value"].keys())[0]]
                except:
                    continue
            
            counts = response["PC_Compounds"][0]["count"] if "count" in response["PC_Compounds"][0] else {}
            for count in counts:
                featurization[count] = counts[count]

            featurization_list.append(featurization)
        except:
            continue
        iter += 1
        print(str(iter))
    df = pd.DataFrame(featurization_list)
    df.to_csv(folder + "mofid_pubchem_features.csv", index=False)
    return df.sort_values(by=["name"])

if __name__ == '__main__':
    folder = sys.argv[1]
    main(PATHNAME + folder + '/')
