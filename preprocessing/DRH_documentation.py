import pandas as pd 
import numpy as np 
from Civ import Civilizations

infile = '../data/raw/drh_20221019.csv'
d = pd.read_csv(infile, low_memory=False) 

#### number of records ####
## raw data
df = d 
len(df['entry_id'].unique()) # 838 (we use this)
len(df['entry_name'].unique()) # 799
len(df['related_q_id'].unique()) # all: 1133

## only super questions
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


# check date ranges 
params = np.loadtxt('humanities-glass/data/mdl_final/cleaned_nrows_455_maxna_5.dat_params.dat')
infile = '/home/vpoulsen/humanities-glass/data/mdl_final/reference_with_entry_id_cleaned_nrows_455_maxna_5.dat'
with open(infile) as f:
    reference = [x.strip() for x in f.readlines()]
reference = [int(x.split()[0]) for x in reference]
d_reference = pd.DataFrame({'entry_id': reference})
d_reference = d_reference.drop_duplicates()

d_year = d[['entry_id', 'entry_name', 'date_range', 'start_year', 'end_year']]

d_year_merge = d_year.merge(d_reference, on = 'entry_id', how = 'inner')
d_year_extreme = d_year_merge.groupby('entry_name').agg({'start_year': 'min', 'end_year': 'max'}).reset_index()
d_year_extreme['start_year'].min() # -4000
d_year_extreme['end_year'].max() # 2025
d_year_extreme['start_year'].mean() # 1076
d_year_extreme['end_year'].mean() # 1422

# entry id 1015 (Ancient Egypt)
d_year_merge[d_year_merge['start_year'] == -4000]
# entry id 751: Buddhism in the Mekong Delta
d_year_merge[d_year_merge['end_year'] == 2025]