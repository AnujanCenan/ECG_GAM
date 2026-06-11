
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path
from pprint import pprint

import sys

def get_required_codes(csv_path, conditions):
    '''
        This function was written by Claude - trivial enough exercise
    '''
    df = pd.read_csv(csv_path)
    
    all_cond_ids = {}

    for idx, row in df.iterrows():
        cond_acronym = row['Acronym Name']

        if cond_acronym in conditions:
            cond_id = row['Snomed_CT']
            all_cond_ids[cond_id] = cond_acronym

    return all_cond_ids

if __name__ == "__main__":
    load_dotenv()
    data_dir = os.getenv("CHAPMAN_DATASET")

    conditions = {}
    if len(sys.argv) > 1:
        for cond in sys.argv[1:]:
            conditions[cond] = []
    else:
        conditions = {'SB':[], 'ST':[], 'SR':[], 'RBBB':[], 'LBBB':[]}
    
    
    cond_ids = get_required_codes(f"{data_dir}/ConditionNames_SNOMED-CT.csv", conditions)
    
    all_records_dir = Path(f"{data_dir}/WFDBRecords")

    for item in all_records_dir.iterdir():
        if not item.is_dir():
            continue

        for child_item in item.iterdir():
            if not child_item.is_dir():
                continue

            for grand_child_item in child_item.iterdir():
                if not grand_child_item.is_file():
                    continue
                file_path = str(grand_child_item)    
                if not file_path.endswith(".hea"):
                    continue

                with open(file_path) as f:
                    for line in f.readlines():
                        if line.startswith("#Dx: "):
                            present_conds = (line.removeprefix("#Dx: ")).split(',')

                            for cond in present_conds:
                                if int(cond) in cond_ids:

                                    record = file_path.removeprefix(data_dir)
                                    conditions[ cond_ids.get(int(cond)) ].append(record)

    for k in conditions.keys():
        for file in conditions[k]:
            print(f"{file},{k}")
