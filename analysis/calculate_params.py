'''
Put parameters into dataframes.
For comparison with macro-level effects.
'''

import numpy as np 
import pandas as pd 
import itertools 

# read data in appropriate format
n_nodes, n_nan, n_rows, n_entries = 20, 5, 455, 407
A = np.loadtxt(f'../data/mdl_experiments/matrix_questions_{n_nodes}_maxna_{n_nan}_nrows_{n_rows}_entries_{n_entries}.txt.mpf_params.dat')
n_J = int(n_nodes*(n_nodes-1)/2)
J = A[:n_J] 
h = A[n_J:]

# wrange hi
nodes = [node+1 for node in range(n_nodes)]
d_nodes = pd.DataFrame(nodes, columns = ['question_id'])
d_nodes['h'] = h

# wrangle Jij
comb = list(itertools.combinations(nodes, 2))
d_edgelist = pd.DataFrame(comb, columns = ['q1', 'q2'])
d_edgelist['Jij'] = J

# save
d_nodes.to_csv('../data/analysis/hi_params.csv', index = False)
d_edgelist.to_csv('../data/analysis/Jij_params.csv', index = False)
