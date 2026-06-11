
import pandas as pd

from dotenv import load_dotenv
import os

def ptbxl_cond_to_ids():
    load_dotenv()
    ptbxl_dir = os.getenv("PTBXL_DATASET")

    records = pd.read_csv(os.path.join(ptbxl_dir, "ptbxl_database.csv"))

    organised_data = {
        'NORM': [],
        'LBBB': [],
        'RBBB': [],
        '1dAVB': []
    }

    for _, record in records.iterrows():
        record_statements = record['scp_codes']
        record_statements = record_statements.removeprefix('{')
        record_statements = record_statements.removesuffix('}')

        record_statements = record_statements.split(',')
        for s in record_statements:
            
            cond, conf = s.split(':')
            cond = cond.replace("\'", "").strip()
            conf = conf.strip()
            ecg_id = record['ecg_id']

            if conf != '100.0':
                continue
            elif cond == r"NORM":
                organised_data['NORM'].append(ecg_id)
                break       # break statement ensures each id goes to exactly one class
            elif cond == r"CRBBB":
                organised_data['RBBB'].append(ecg_id)
                break
            elif cond == r"CLBBB":
                organised_data['LBBB'].append(ecg_id)
                break
            elif cond == r"1AVB":
                organised_data['1dAVB'].append(ecg_id)
                break
    return organised_data

if __name__ == "__main__":
    organised_data = ptbxl_cond_to_ids()
    for cond in organised_data.keys():
        print(f"===== {cond} =====")
        for ecg_id in organised_data[cond]:
            print(ecg_id)
        print()


    for cond1 in organised_data.keys():
        for cond2 in organised_data.keys():
            if cond1 >= cond2:
                continue

            intersection = (set.intersection(set(organised_data[cond1]), set(organised_data[cond2])))
            
            print(f"intersection between {cond1} and {cond2} == {len(intersection)}")



    



