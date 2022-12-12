import pandas as pd 
import numpy as np 

# setup
n_rows, n_nan, n_nodes, n_top_states = 455, 5, 20, 150

# our files 
df_v = pd.read_csv('../data/reference/main_nrow_561_ncol_21_nuniq_20_suniq_491_maxna_5.csv')
mat_v = np.loadtxt('../data/clean/matrix_nrow_561_ncol_21_nuniq_20_suniq_491_maxna_5.txt')

# simon files 
infile = f'../data/mdl_final/reference_with_entry_id_cleaned_nrows_455_maxna_5.dat'
with open(infile) as f:
    reference = [x.strip() for x in f.readlines()]
reference = [int(x.split()[0]) for x in reference]
df_s = pd.DataFrame({'entry_id': reference})

## which files do Simon have currently. 
df_s = df_s.drop_duplicates()
df_s # 407

## which files do we have currently. 
df_v = df_v[['s']].drop_duplicates()
df_v # 491 

## how can we tell the difference? 
d_main = pd.read_csv()