import numpy as np 
import pandas as pd 
from scipy.spatial.distance import hamming
from sim_fun import p_dist, bin_states
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import networkx as nx
from itertools import combinations, product
from sklearn.manifold import MDS
import time 

## NB: this code is pretty slow, so we should consider 
## what we can do to optimize it 
## In particular, it scales very poorly with number of nodes

# functions
def compute_HammingDistance(X):
    return (X[:, None, :] != X).sum(2)

# setup
seed = 2
np.random.seed(seed)
# sample data
n_nodes = 17
scale_h = 1
scale_J = 1
h = np.random.normal(scale=scale_h, size=n_nodes)
J = np.random.normal(scale=scale_J, size=n_nodes*(n_nodes-1)//2)
x0 = time.time()
p = p_dist(h, J)
x1 = time.time()
allstates = bin_states(n_nodes) 
x2 = time.time() 

# subset the states we want 
p_cutoff = np.mean(p)
x3 = time.time()
p_ind = [i for i,v in enumerate(p) if v > p_cutoff]
x4 = time.time()
substates = allstates[p_ind]

# for scaling later
p_vals = p[p > p_cutoff]

# run it on these substates
x5 = time.time()
distances = compute_HammingDistance(substates) # here typically euclidean
x6 = time.time()

x7 = time.time()
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
x8 = time.time()

# networkx is not super smart so we need to do some stuff here
states = [x for x in range(len(p_ind))]
x9 = time.time()
comb = list(combinations(states, 2))
x10 = time.time()

# create network
x11 = time.time()
G = nx.Graph(comb)
x12 = time.time()

# terrible
print(len(substates))
print(f'0: {x1-x0}') # this is where we die (p_dist needs to be made more effective)
print(f'1: {x2-x1}') # allstates is hard (possibly MCMC)
print(f'2: {x4-x3}')
print(f'3: {x6-x5}')
print(f'4: {x8-x7}') # MDS is hard (has to move if insufficient)
print(f'5: {x10-x9}')
print(f'6: {x12-x11}')
print(f'total = {x12-x1}')

# n = 16, s = 155, 38.9s, 1.19s, 25.8s
# n = 17, s = 97, 118.6s, 2.5s, 0.66s 

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
fig, ax = plt.subplots()
nx.draw_networkx_nodes(G, pos, node_size = size_lst)

# see: https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html#sphx-glr-auto-examples-manifold-plot-mds-py

### now do it on real data (wait for Simon Jij, hi) ###

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