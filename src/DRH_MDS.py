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
import glob
import re
import time 

# setup
outpath = '../fig/MDS'
type = 'sim'
seed = 2
n_nodes = 20
scale_h = 1
scale_J = 1
n_cutoff = 500

# sample data
np.random.seed(seed)
h = np.random.normal(scale=scale_h, size=n_nodes)
J = np.random.normal(scale=scale_J, size=n_nodes*(n_nodes-1)//2)
p = p_dist(h, J)
allstates = bin_states(n_nodes) 

# subset the states we want 
val_cutoff = np.sort(p)[::-1][n_cutoff]
p_ind = [i for i,v in enumerate(p) if v > val_cutoff]
p_vals = p[p > val_cutoff]
substates = allstates[p_ind]
perc = round(np.sum(p_vals)*100,2) 

# run it on these substates
distances = compute_HammingDistance(substates) # here typically euclidean

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
out = os.path.join(outpath, f'type_{type}_nnodes_{n_nodes}_hscale_{scale_h}_Jscale_{scale_J}_ncutoff_{n_cutoff}_perc_{perc}_seed_{seed}.jpeg')
fig, axis = plt.subplots(facecolor = 'w', edgecolor = 'k')
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size = size_lst)
plt.savefig(f'{out}')

# see: https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html#sphx-glr-auto-examples-manifold-plot-mds-py

### now do it on real data (wait for Simon Jij, hi) ###
### why does this take so much longer ...?
outpath = '../fig/MDS'
inpath = '../data/mdl'
filetype = '*.txt'
globpath = os.path.join(inpath, filetype)
filelst = glob.glob(globpath)
file = filelst[0]
regex_pattern = 'nodes_(\d+)_maxna_(\d+)'
p = re.search(regex_pattern, file)
n_nodes, maxna = int(p[1]), int(p[2])

type = 'DRH'
seed = 2
n_cutoff = 500

## load stuff
nJ = int(n_nodes*(n_nodes-1)/2)
A = np.loadtxt(file, delimiter = ',')
J = A[:nJ]
h = A[nJ:]
                        
p = p_dist(h, J) # this was okay actually
allstates = bin_states(n_nodes) 

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
out = os.path.join(outpath, f'type_{type}_nnodes_{n_nodes}_maxna_{maxna}_ncutoff_{n_cutoff}_perc_{perc}_seed_{seed}.jpeg')
fig, axis = plt.subplots(facecolor = 'w', edgecolor = 'k')
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size = size_lst)
plt.savefig(f'{out}')

''' xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx '''

### now trace a line ###
## this seems much harder

### now do conditional sampling ###
# dummy example 
data = np.array([[1, -1], [-1, 1]])
probs = np.array([0.35, 0.05, 0.45, 0.1])

### conditional probability where only one state is left unanswered
# this needs to be generalized to n blanks 
# using p, allstates from above currently
l = [[1, -1], [1], [1], [1], [1]]
def compute_ratio(l, allstates): 
    N = len(l)
    marginal_states = np.array([p for c in combinations(l, N) for p in product(*c)])
    marginal_both = np.array(np.all((marginal_states[:,None,:]==allstates[None,:,:]),axis=-1).nonzero()).T.tolist()
    marginal_orig = [y for x, y in marginal_both]
    probs = p[marginal_orig]
    x1, x2 = probs
    prob_x1 = x1/(x1+x2)*100
    prob_x2 = x2/(x1+x2)*100
    return [prob_x1, prob_x2], allstates[marginal_orig]

probabilities, states = compute_ratio(l, allstates)


## optimize

seed = 2
np.random.seed(seed)
# sample data
n_nodes = 10
scale_h = 1
scale_J = 1
h = np.random.normal(scale=scale_h, size=n_nodes)
J = np.random.normal(scale=scale_J, size=n_nodes*(n_nodes-1)//2)
p = p_dist(h, J)
p2 = p_dist2(h, J) 
## double check that it gives the same ##