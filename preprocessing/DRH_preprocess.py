'''
NB: when we have a clean pipeline we should come back here
and only select the columns we actually need.
Potentially rename e.g. related_q_id to q_id, if we do end
up using only that. 

data/x/drh_xxx.json
'''

import pandas as pd 
import argparse 
import json
import re

def load_file(infile):
    with open(infile, "r") as f: 
        data = json.load(f)
    return data

def json_to_dct(data):
    dct = {
        "related_q_id": [row["Related question ID"] for row in data], # clean
        "related_q": [row["Related Question"] for row in data], # clean 
        "related_parent_q": [row["Related Parent Question"] for row in data], # clean
        "poll": [row["Poll"] for row in data],
        "q": [row["Question"] for row in data],
        "q_id": [row["Question ID"] for row in data],
        "answers": [row["Answers"] for row in data],
        "answer_val": [row["Answer values"] for row in data],
        "note": [row["Note"] for row in data],
        "parent_answer": [row["Parent answer"] for row in data],
        "parent_question": [row["Parent question"] for row in data],
        "parent_answer_val": [row["Parent answer value"] if has_att(row, "Parent answer value") else "MISSING" for row in data],
        "entry_name": [row["Entry name"] for row in data],
        "entry_id": [row["Entry ID"] for row in data],
        "entry_desc": [row["Entry description"] for row in data],
        "date_range": [row["Date range"] for row in data],
        "start_year": [row["start_year"] for row in data],
        "end_year": [row["end_year"] for row in data],
        "branching_q": [row["Branching question"] for row in data],
        "entry_src": [row["Entry source"] for row in data],
        "entry_tags": [row["Entry tags"] for row in data],
        "region_name": [row["Region name"] for row in data],
        "region_id": [row["Region ID"] for row in data],
        "region_desc": [row["Region description"] for row in data],
        "region_tags": [row["Region tags"] for row in data],
        "expert": [row["Expert"] for row in data],
    }
    return dct

def has_att(row, att): 
    has_attribute = True
    try: 
        row[att]
    except KeyError: 
        has_attribute = False 
    return has_attribute

def main(inpath):
    # read file
    json_raw = load_file(inpath) # data/x/drh_xxx.json
    outpath = re.sub('.json', '.csv', inpath)
    # extract the data 
    dct_raw = json_to_dct(json_raw)
    df_raw = pd.DataFrame(dct_raw)
    # write file 
    df_raw.to_csv(outpath, index = False) # data/y/DRH_raw.csv

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inpath", required=True, type=str, help="path to input file (pickle)")
    args = vars(ap.parse_args())
    main(inpath = args['inpath'])