import numpy as np 
import pandas as pd 
from scipy.spatial.distance import hamming
from sim_fun import p_dist, bin_states, compute_HammingDistance
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import networkx as nx
import itertools
from sklearn.manifold import MDS
import os 
pd.set_option('display.max_colwidth', None)

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

## this can wait for the next document
allstates = bin_states(n_nodes) # this takes some time (and should not attempted with n_nodes > 20)

# save these files 
p.shape
allstates.shape

# subset the states we want 
val_cutoff = np.sort(p)[::-1][n_cutoff]
p_ind = [i for i,v in enumerate(p) if v > val_cutoff]
p_vals = p[p > val_cutoff]
substates = allstates[p_ind]
perc = round(np.sum(p_vals)*100,2) 

# run it on these substates
distances = compute_HammingDistance(substates) # here typically euclidean

## this takes a while for our data
## probably because we do not hit epsilon
## also takes up a HUGE amount of memory
## probably we include too many possible states
mds = MDS(
    n_components = 2,
    n_init = 4, # number initializations of SMACOF alg. 
    max_iter = 300, # set up after testing
    eps=1e-3, # set up after testing
    random_state=seed,
    dissimilarity='precomputed',
    n_jobs=-1 # check this if it is slow 
)
pos = mds.fit(distances).embedding_ # understand fit, fit_transform

# networkx is not super smart so we need to do some stuff here
states = [x for x in range(len(p_ind))]
comb = list(itertools.combinations(states, 2))

# create network
G = nx.Graph(comb)

# add node attributes
d_nodes = pd.DataFrame(
    list(zip(p_ind, p_vals)),
    columns = ['label', 'size']
)

dct_nodes = d_nodes.to_dict('index')

labeldict = {}
for key, val in dct_nodes.items():
    G.nodes[key]['size'] = val['size']
    labeldict[key] = val['label']

# prepare plot
node_scaling = 2000
size_lst = list(nx.get_node_attributes(G, 'size').values())
size_lst = [x*node_scaling for x in size_lst]

# plot 
out = os.path.join(outpath, f'type_{filetype}_nnodes_{n_nodes}_maxna_{maxna}_ncutoff_{n_cutoff}_perc_{perc}_seed_{seed}.jpeg')
fig, axis = plt.subplots(facecolor = 'w', edgecolor = 'k')
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size = size_lst)
plt.savefig(f'{out}')