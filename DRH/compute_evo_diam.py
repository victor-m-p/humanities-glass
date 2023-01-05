import pandas as pd 
import matplotlib.pyplot as plot 
import seaborn as sns 
import numpy as np 

# usual setup
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')

# generate all states
n_nodes = 20
from fun import bin_states 
configurations = bin_states(n_nodes) 

# read the curated data 
d_edgelist = pd.read_csv('../data/COGSCI23/evo/overview.csv')

# expand in a different way ...
## okay, first just for n = 10
d_edgelist_10 = d_edgelist[d_edgelist['t_to'] == 10]


d_edgelist.head(20)

x = d_edgelist_10.groupby(['config_from'])['config_to'].unique().reset_index(name = 'attractors')



y = d_edgelist_10.groupby(['config_from', 'config_to']).size().reset_index(name = 'weight')
y.groupby('config_from')['']

# test 
att = x['attractors'].tolist()
att = att[0]

def hamming_distance(X):
    '''https://stackoverflow.com/questions/42752610/python-how-to-generate-the-pairwise-hamming-distance-matrix'''
    return (X[:, None, :] != X).sum(2)

configurations

h_att = hamming_distance(att)


d_edgelist_w = d_edgelist.groupby(['config_from', 'config_to', 't_to']).size().reset_index(name = 'weight')

