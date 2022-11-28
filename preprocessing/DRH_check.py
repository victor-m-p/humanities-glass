import numpy as np 
import pandas as pd
pd.set_option('display.max_colwidth', None)

# read data
d_full = pd.read_csv('data/DRH/clean/nref_nrow_201_ncol_21_nuniq_20_suniq_181_tol_0.0.csv')
d_na = pd.read_csv('data/DRH/clean/nref_nrow_605_ncol_21_nuniq_20_suniq_530_tol_0.35.csv')

# sanity check, d_full:
p_full = pd.read_csv('data/DRH/clean/pandas_nrow_201_ncol_21_nuniq_20_suniq_181_tol_0.0.csv')
pcols = p_full.columns
pcols = pcols[1:-1]
dref = pd.DataFrame(pcols, columns = {'related_q_id'})
dref['id'] = dref.index
dref['related_q_id'] = [int(x) for x in dref['related_q_id']]

# dfull
df = d_full.merge(dref, on = 'related_q_id')
cols = [9, 15, 16, 17]
negs = df[df['id'].isin(cols)]
negs # why is sacrifice of children not the most uncommon?

# sanity check, d_na: 
p_na = pd.read_csv('data/DRH/clean/pandas_nrow_605_ncol_21_nuniq_20_suniq_530_tol_0.35.csv')
pcols = p_full.columns
pcols = pcols[1:-1]
dref = pd.DataFrame(pcols, columns = {'related_q_id'})
dref['id'] = dref.index
dref['related_q_id'] = [int(x) for x in dref['related_q_id']]

# dna 
da = d_na.merge(dref, on = 'related_q_id')
cols = [18, 19]
negs = da[da['id'].isin(cols)]
negs # here it makes sense (castration)

# check big gods clump. 

# check timing with 0, 12 coupling (early, late). 