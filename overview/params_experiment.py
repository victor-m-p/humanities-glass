import numpy as np 
import pandas as pd 

n_nodes, n_nan, n_rows, n_entries = 20, 5, 455, 407
basepath = f'../data/mdl_experiments/matrix_questions_{n_nodes}_maxna_{n_nan}_nrows_{n_rows}_entries_{n_entries}'
params_hidden = np.loadtxt(f'{basepath}.txt.mpf_HIDDEN_params.dat')
params_removed = np.loadtxt(f'{basepath}.txt.mpf_REMOVED_params.dat')
params_original = np.loadtxt(f'{basepath}.txt.mpf_params.dat') 
params_added = np.loadtxt(f'{basepath}.txt.mpf_ADDED_params.dat')

# PLOT PARAMETERS 


n_J = int(n_nodes*(n_nodes-1)/2)
J = A[:n_J] 
h = A[n_J:]