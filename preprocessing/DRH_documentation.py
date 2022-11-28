import pandas as pd 
import numpy as np 
from Civ import Civilizations

infile = '../data/raw/drh_20221019.csv'
df = pd.read_csv(infile, low_memory=False) 

df2 = df[df['related_parent_q'].isna()] 

## raw data
len(df['entry_id'].unique()) # 838 (we use this)
len(df['entry_name'].unique()) # 799
len(df['related_q_id'].unique()) # all: 1133

## only super
df = df[df['related_parent_q'].isna()] 
len(df['entry_id'].unique()) # 838
len(df['entry_name'].unique()) # 799
len(df['related_q_id'].unique()) # super only: 171

## binary only 
df = df[["entry_id", "entry_name", "related_q_id", "related_q", "answers"]]  
df.replace(
    {'answers': 
        {"Field doesn't know": 'Unknown', 
        "I don't know": 'Unknown'}},
    inplace = True)
civ = Civilizations(df)
civ.preprocess()

dfuniq = civ.duniq
len(dfuniq['related_q_id'].unique()) # 149
len(dfuniq['entry_id'].unique()) # 835
dfuniq['has_answer'].mean()

# run through specific params
n_questions = 20
n_nan = 4
civ.set_constraints(n_questions, n_nan/n_questions, "related_q_id")
civ.n_best()
civ.max_constraints()
civ.max_tolerance()
civ.weight_format()

dcsv = civ.dcsv
len(dcsv['s'].unique()) # 471
