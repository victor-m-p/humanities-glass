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
pd.set_option('display.max_colwidth', None)

#inpath = '../data/mdl'
#filetype = '*.txt'
#globpath = os.path.join(inpath, filetype)
#filelst = glob.glob(globpath)
#file = filelst[0]
#regex_pattern = 'nodes_(\d+)_maxna_(\d+)'
#p = re.search(regex_pattern, file)
#n_nodes, maxna = int(p[1]), int(p[2])

file = '/home/vpoulsen/humanities-glass/analysis/pars/matrix_nrow_541_ncol_21_nuniq_20_suniq_471_maxna_4.txt.mpf_params_NN1_LAMBDA0.411147'
n_nodes, maxna = 20, 4
outpath = '../fig/MDS'

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

####### for specific ratio ########
def compute_ratio(p, l, allstates): 
    N = len(l)
    marginal_states = np.array([p for c in itertools.combinations(l, N) for p in itertools.product(*c)])
    marginal_both = np.array(np.all((marginal_states[:,None,:]==allstates[None,:,:]),axis=-1).nonzero()).T.tolist()
    marginal_orig = [y for x, y in marginal_both]
    probs = p[marginal_orig]
    probs_norm = [float(i)/sum(probs) for i in probs]
    return probs_norm, allstates[marginal_orig]

# load dcsv
d = pd.read_csv('../data/reference/main_nrow_541_ncol_21_nuniq_20_suniq_471_maxna_4.csv')
nref = pd.read_csv('../data/reference/nref_nrow_541_ncol_21_nuniq_20_suniq_471_maxna_4.csv')
sref = pd.read_csv('../data/reference/sref_nrow_541_ncol_21_nuniq_20_suniq_471_maxna_4.csv')

# find duplicates (with no NAN)
allcols = d.columns
qcols = allcols[1:-1]

dClean = d.loc[~(d==0).any(axis=1)]

# 190 = Sri Lankan Buddhism
## small-scale rituals
## large-scale rituals
dSL = d.loc[d.s == 190, qcols].values.tolist()
r1, r2, r3, r4 = dSL[0], dSL[1], dSL[2], dSL[3]
## bad manual solution for now 
l = []
for a1, a2, a3, a4 in zip(r1, r2, r3, r4): 
    lr = [a1, a2, a3, a4]
    l.append([*set(lr)])

p_norm, states = compute_ratio(p, l, allstates)
p_norm
states
sref.head(10)
nref.tail(5)
## 71.53%: +small +large
## 4.21%: +small -large
## 14.40: -small +large
## 9.85$: -small -large