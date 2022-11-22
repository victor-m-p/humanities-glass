import numpy as np 
import pandas as pd

# read data
d_full = pd.read_csv('data/DRH/clean/nref_nrow_201_ncol_21_nuniq_20_suniq_181_tol_0.0.csv')
d_na = pd.read_csv('data/DRH/clean/nref_nrow_605_ncol_21_nuniq_20_suniq_530_tol_0.35.csv')

# id should match node name  
d_na['id'] = d_na.index
d_full['id'] = d_full.index 

# full 