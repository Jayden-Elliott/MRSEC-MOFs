import os
import sys
import pandas as pd
from mofid.run_mofid import cif2mofid
from rdkit.Chem.Descriptors import *
import urllib.parse
import requests
import re
import time

PATHNAME = os.path.abspath('.') + '/'


def main(folder):
    print("Generating MOFid and PubChem features...")
    featurization_list = []
    iter = 0
    all_feats = pd.read_csv("/Users/jayden/Documents/Code/MOFS_MRSEC/data/kulik/all_features.csv")
    last = time.time()
    for name in all_feats["name"]:
        try:
            if time.time() - last < 0.4:
                time.sleep(0.4 - (time.time() - last))
            last = time.time()

            featurization = {"name": name}
            smiles = all_feats["Canonical-SMILES"][all_feats["name"] == name].values[0]
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{urllib.parse.quote(smiles)}/record/JSON"
            response = requests.get(url).json()
            
            cid = ""
            try:
                cid = response["PC_Compounds"][0]["id"]["id"]["cid"]
            except:
                featurization_list.append(featurization)
                continue

            exp_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Experimental+Properties"
            exp_response = requests.get(exp_url).json()

            if exp_response.get("Record") is None:
                featurization_list.append(featurization)
                continue

            exp_props = exp_response["Record"]["Section"][0]["Section"][0]["Section"]
            
            for prop in exp_props:
                for temp_prop in ["Melting Point", "Boiling Point", "Flash Point"]:
                    if prop["TOCHeading"] == temp_prop:
                        t_sum = 0
                        t_count = 0
                        for datum in prop["Information"]:
                            try:
                                if datum["Value"].get("StringWithMarkup") is None:
                                    continue
                                t_string = re.sub(r"\s+", "", datum["Value"]["StringWithMarkup"][0]["String"])
                                if re.match(r"[0-9.]*", t_string[:t_string.find("°")]):
                                    if "C" == t_string[t_string.find("°") + 1]:
                                        t_sum += float(t_string[:t_string.find("°")])
                                        t_count += 1
                                    elif "F" == t_string[t_string.find("°") + 1]:
                                        t_sum += (float(t_string[:t_string.find("°")]) - 32) * 5 / 9
                                        t_count += 1
                            except:
                                continue
                        featurization[temp_prop.replace(" ", "-")] = t_sum / t_count if t_count > 0 else 0

                if prop["TOCHeading"] == "Density":
                    denstiy_sum = 0
                    density_count = 0
                    for datum in prop["Information"]:
                        try:
                            if datum["Value"].get("StringWithMarkup") is None:
                                continue
                            t_string = re.sub(r"\s+", "", datum["Value"]["StringWithMarkup"][0]["String"])
                            if re.match(r"[0-9.]*", t_string[:t_string.find("g/cm")]) or re.match(r"[0-9.]*", t_string):
                                denstiy_sum += float(t_string[:t_string.find("g/cm")]) if "g/cm" in t_string else float(t_string)
                        except:
                            continue
                    featurization["Density"] = denstiy_sum / density_count if density_count > 0 else 0

                if prop["TOCHeading"] == "Dissociation Constants":
                    diss_sum = 0
                    diss_count = 0
                    for datum in prop["Information"]:
                        try:
                            if datum["Value"].get("StringWithMarkup") is None:
                                continue
                            t_string = re.sub(r"\s+", "", datum["Value"]["StringWithMarkup"][0]["String"])
                            if re.match(r"[0-9.]*", t_string):
                                denstiy_sum += float(t_string)
                        except:
                            continue
                    featurization["Dissociation Constants"] = diss_sum / diss_count if diss_count > 0 else 0

            featurization_list.append(featurization)
        except:
            continue
        iter += 1
        print(str(iter))
    df = pd.DataFrame(featurization_list)
    df.to_csv(folder + "pubchem_exp_features.csv", index=False)
    return df.sort_values(by=["name"])

if __name__ == '__main__':
    folder = sys.argv[1]
    main(PATHNAME + folder + '/')
