'''
Preprocessing: 
- load CPT code descriptions
- load HCPCS code descriptions
- CPT code descriptions and HCPCS code descriptino mapping will be used in retrieval
'''
import argparse
import json
import os
import sqlite3
from typing import Dict, List, Set, Tuple
import pandas as pd

def load_cpt_descriptions(cpt_desc_path: str) -> Dict[str, str]:
    """Load CPT code from excel , return a dict mapping code to description"""
    cpt_description = {}
    cpt_df = pd.read_excel(cpt_desc_path, sheet_name="CPT")
    for _, row in cpt_df.iterrows():
        code = str(row['CPTCd']).strip()
        desc = str(row['FullDesc']).strip()
        cpt_description[code] = desc

    hcpcs_df = pd.read_excel(cpt_desc_path, sheet_name="HCPCS")
    for _, row in hcpcs_df.iterrows():
        code = str(row['HCPCS_CODE']).strip()
        desc = str(row['FULL_DESCRIPTION']).strip()
        cpt_description[code] = desc
    return cpt_description