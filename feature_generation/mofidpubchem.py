import os
import sys
import pandas as pd
from mofid.run_mofid import cif2mofid
from rdkit.Chem.Descriptors import *
import urllib.parse
import requests
import re

PATHNAME = os.path.abspath('.') + '/'


def main(folder):
    print("Generating MOFid and PubChem features...")
    featurization_list = []
    iter = 0
    for filename in os.listdir(folder + "cifs/"):
        response = ""
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
    df.to_csv(folder + "mofid_pubchem_features.csv", index=False)
    return df.sort_values(by=["name"])

if __name__ == '__main__':
    folder = sys.argv[1]
    main(PATHNAME + folder + '/')
