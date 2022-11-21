import pandas as pd 
from Civ import Civilizations

# read data
df = pd.read_csv("data/raw/df_raw.csv")
df = df[["entry_id", "entry_name", "related_q_id", "related_q", "related_parent_q", "answers"]] 

# sort-of works as intended now 
N = 20
for N_NAN in (0, 7): # 0 NAN and 7 NAN
    civ = Civilizations(df)
    civ.preprocess() # can only be run once (fixed)
    civ.set_constraints(20, N_NAN/N, "related_q_id") # when we run this
    civ.n_best()
    civ.max_constraints()
    civ.max_tolerance()
    civ.weight_format()
    civ.write_data('data/weighted')