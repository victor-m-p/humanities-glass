import pandas as pd 
import numpy as np 
from fun import p_dist, bin_states

# setup
n_nodes = 20
n_nan = 5
n_rows = 455
n_entries = 407

params = np.loadtxt(f'../data/mdl_experiments/matrix_questions_{n_nodes}_maxna_{n_nan}_nrows_{n_rows}_entries_{n_entries}.txt.mpf_params.dat')
nJ = int(n_nodes*(n_nodes-1)/2)
J = params[:nJ]
h = params[nJ:]
p = p_dist(h, J) # takes a minute (and a lot of memory). 
np.savetxt(f'../data/preprocessing/p_new.txt', p)

params = np.loadtxt('../data/mdl_original/cleaned_nrows_455_maxna_5.dat_params.dat')
nJ = int(n_nodes*(n_nodes-1)/2)
J = params[:nJ]
h = params[nJ:]
p = p_dist(h, J)
np.savetxt(f'../data/preprocessing/p_old.txt', p)

configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype=int)
p_new = np.loadtxt('../data/preprocessing/p_new.txt')
p_old = np.loadtxt('../data/preprocessing/p_old.txt')
p_humanities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
p_cognitive = np.loadtxt('../../co')
p_cultural = np.loadtxt()

