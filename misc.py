import os
import sys
import pandas as pd
import requests
import json
import urllib.parse

PATHNAME = os.path.abspath('.') + '/'

smiles = input().replace("O-", "OH")


url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{urllib.parse.quote(smiles)}/record/JSON"

print(url)
response = requests.get(url).json()

features = {}

blocklist = ["Fingerprint", "IUPAC Name", "InChI", "Molecular Formula", "SMILES"]
props = response["PC_Compounds"][0]["props"]
for prop in props:
    skip = False
    for block in blocklist:
        if block in prop["urn"]["label"]:
            skip = True
    if skip: continue

    feat_name = (prop["urn"]["name"] if "name" in prop["urn"] else "" + prop["urn"]["label"]).replace(" ", "-")
    features[feat_name] = prop["value"][next(iter(prop["value"]))]

counts = response["PC_Compounds"][0]["count"]
for count in counts:
    features[count] = counts[count]

print(features)