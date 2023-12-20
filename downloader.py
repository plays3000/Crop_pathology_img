import sys
import os
from tqdm import tqdm

"""
This code is for downloading TCGA open dataset from https:///api.gdc.cancer.gov.

If there are some dataset you want to download, please download menifest.txt file in the website.

And read clinical_csv_maker.ipynb first before conducting this codes.
"""

with open('menifest.txt','r') as f:
    column = f.readline()
    while True:
        line = f.readline()
        if not line:
            break
        try:
            lines = line.split('	')
            id = lines[0]
            filename = lines[1]
            args = f'curl --remote-name --remote-header-name https://api.gdc.cancer.gov/data/{id}'
            print(f"Downloading {filename} is getting start!")
            os.system(args)
            print(f"Downloading {filename} is completed!")
            continue
        except:
            print(f"Downloading {filename} is NOT completed!")
            pass