import numpy as np 
import pandas as pd 
from scipy.spatial.distance import hamming
from sim_fun import p_dist, bin_states
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import networkx as nx
from itertools import combinations, product
from sklearn.manifold import MDS

# functions
def compute_HammingDistance(X):
    return (X[:, None, :] != X).sum(2)

# setup
seed = 2
np.random.seed(seed)

# sample data
n_nodes = 5
scale_h = 0.5
scale_J = 1
h = np.random.normal(scale=scale_h, size=n_nodes, )
J = np.random.normal(scale=scale_J, size=n_nodes*(n_nodes-1)//2)
p = p_dist(h, J)
allstates = bin_states(n_nodes) 

# subset the states we want 
p_cutoff = np.median(p)
p_ind = [i for i,v in enumerate(p) if v > p_cutoff]
substates = allstates[p_ind]

# for scaling later
p_vals = p[p > p_cutoff]

# run it on these substates
distances = compute_HammingDistance(substates) # here typically euclidean

mds = MDS(
    n_components = 2,
    n_init = 4, # number initializations of SMACOF alg. 
    max_iter = 10000, # set down if slow (default = 300)
    eps=1e-12, # set down if slow (default = 1e-3)
    random_state=seed,
    dissimilarity='precomputed',
    n_jobs=1 # check this if it is slow 
)
pos = mds.fit(distances).embedding_

# networkx is not super smart so we need to do some stuff here
states = [x for x in range(len(p_ind))]
comb = list(combinations(states, 2))
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
fig, ax = plt.subplots()
nx.draw_networkx_nodes(G, pos, node_size = size_lst)

### now do it on real data ### 

# https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html#sphx-glr-auto-examples-manifold-plot-mds-py

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

