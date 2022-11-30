import numpy as np 
from sim_fun import p_dist, bin_states
import pandas as pd 

# setup
infile = '../data/analysis/matrix_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.txt.mpf_params_NN1_LAMBDA0.453839'
d_main = '../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'

# collapse weighted rows to nan
outname = '../data/analysis/d_collapsed_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.csv'
d_main = pd.read_csv(d_main)
qcols = d_main.columns[1:-1]
d_unweighted = d_main.groupby('s')[qcols].mean().reset_index().astype(int)
d_unweighted.to_csv(outname, index = False)

# calculate probability of all configurations based on parameters h, J.
outname = '../data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt'
n_nodes = 20
nJ = int(n_nodes*(n_nodes-1)/2)
A = np.loadtxt(infile, delimiter = ',')
J = A[:nJ]
h = A[nJ:]
p = p_dist(h, J) # this takes some time (and should not be attempted with n_nodes > 20)
np.savetxt(outname, p)

# allstates (for julia)
outname = '../data/analysis/allstates_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt'
allstates = bin_states(n_nodes) # takes a minute (do not attempt with n_nodes > 20)
np.savetxt(outname, allstates)