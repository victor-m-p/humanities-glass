import numpy as np 
from sim_fun import p_dist

# setup
infile = '../data/analysis/matrix_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.txt.mpf_params_NN1_LAMBDA0.453839'
outname = '../data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt'
n_nodes = 20
nJ = int(n_nodes*(n_nodes-1)/2)
A = np.loadtxt(infile, delimiter = ',')
J = A[:nJ]
h = A[nJ:]

# calculate probability of all configurations based on parameters h, J.
p = p_dist(h, J) # this takes some time (and should not be attempted with n_nodes > 20)

# save this 
np.savetxt(outname, p)