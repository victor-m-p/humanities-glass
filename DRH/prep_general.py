'''
VMP 2022-12-12: 
Prepares key documents for the analysis of DRH data. 
'''

import numpy as np 
from fun import p_dist, bin_states, top_n_idx, hamming_distance
import pandas as pd 
from sklearn.manifold import MDS

# setup
n_nodes = 20
n_nan = 5

# important reference for now (only states that have these ids)
infile = f'../data/mdl_final/reference_with_entry_id_cleaned_nrows_455_maxna_{n_nan}.dat'
with open(infile) as f:
    reference = [x.strip() for x in f.readlines()]
reference = [int(x.split()[0]) for x in reference]
d_reference = pd.DataFrame({'entry_id': reference})
d_reference = d_reference.drop_duplicates()

# recreate the node reference document
nodes_reference = pd.read_csv('../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv')
nodes_reference = nodes_reference.merge(d_reference, on = 'entry_id', how = 'inner')
nodes_reference.to_csv(f'../data/analysis/nref_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.csv', index = False)
states_reference = pd.read_csv('../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv')
states_reference.to_csv(f'../data/analysis/sref_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.csv', index = False)

# collapse weighted rows to nan
d_main = pd.read_csv('../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv')
d_main = d_main.rename(columns = {'s': 'entry_id'})
d_main = d_main.merge(d_reference, on = 'entry_id', how = 'inner')
qcols = d_main.columns[1:-1]
d_unweighted = d_main.groupby('entry_id')[qcols].mean().reset_index().astype(int)
d_unweighted.to_csv(f'../data/analysis/d_collapsed_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.csv', index = False)
d_main.to_csv(f'../data/analysis/d_main_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.csv', index = False)

# calculate probability of all configurations based on parameters h, J.
params = np.loadtxt('../data/mdl_final/cleaned_nrows_455_maxna_5.dat_params.dat')
n_nodes = 20
nJ = int(n_nodes*(n_nodes-1)/2)
J = params[:nJ]
h = params[nJ:]
p = p_dist(h, J) # this takes some time (and should not be attempted with n_nodes > 20)
np.savetxt(f'../data/analysis/p_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.txt', p)

# allstates
allstates = bin_states(n_nodes) # takes a minute (do not attempt with n_nodes > 20)
np.savetxt(f'../data/analysis/allstates_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.txt', allstates.astype(int), fmt='%i')