import numpy as np 
import pandas as pd 
from sim_fun import bin_states, compute_HammingDistance
from sklearn.manifold import MDS
import itertools
import networkx as nx 
import matplotlib.pyplot as plt
import os 

# setup 
n_nodes, maxna = 20, 10
seed = 2
n_cutoff = 500
outpath = '../fig'

# read files 
p_file = '../data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt'
d_file = '../data/clean/matrix_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.txt'
p = np.loadtxt(p_file)
datastates_weighted = np.loadtxt(d_file)
datastates = np.delete(datastates_weighted, n_nodes, 1)
datastates_uniq = np.unique(datastates, axis = 0)

# get all state configurations 
allstates = bin_states(n_nodes) # takes a minute (do not attempt with n_nodes > 20)

# subset states above threshold
val_cutoff = np.sort(p)[::-1][n_cutoff]
p_ind = [i for i,v in enumerate(p) if v > val_cutoff]
p_vals = p[p > val_cutoff]
substates = allstates[p_ind]
perc = round(np.sum(p_vals)*100,2) 

# compute hamming distance
distances = compute_HammingDistance(substates) 

# set this up when we have a plot we like
mds = MDS(
    n_components = 2,
    n_init = 4, # number initializations of SMACOF alg. 
    max_iter = 300, # set up after testing
    eps=1e-3, # set up after testing
    random_state=seed,
    dissimilarity='precomputed',
    n_jobs=-1 # check this if it is slow 
)
pos = mds.fit(distances).embedding_ # check fit, fit_transform

# couple configurations, data states and ids
substates_id = [(num, ele) for num, ele in enumerate(substates)]
substates_id = pd.DataFrame(substates_id, columns = ['index', 'config']) # this is good
comb = list(itertools.combinations(substates_id['index'].values, 2))

# find the configurations that are observed in our data
state_overlap = np.array([x for x in set(tuple(x) for x in substates) & set(tuple(x) for x in datastates_uniq)])
state_overlap = pd.DataFrame(
    [(num, ele) for num, ele in enumerate(state_overlap)],
    columns = ['id_overlap', 'config']
)

# notably, only 79 ...
# perhaps we should keep track of the data states 
# that do not overlap. 
# interesting that some (unobserved) religions 
# are more plausible than actually observed religions
state_overlap_str = state_overlap.astype({'config': str})
substates_id_str = substates_id.astype({'config': str})
node_attr = substates_id_str.merge(state_overlap_str, on = 'config', how = 'left', indicator = True)
node_attr = node_attr.rename(columns = {'_merge': 'datastate'})
node_attr.replace(
    {'datastate': 
        {'both': 'Yes', 'left_only': 'No'}},
    inplace = True)
node_attr = node_attr[['index', 'datastate']]
node_attr['label'] = p_ind
node_attr['size'] = p_vals

# create network
G = nx.Graph(comb)

dct_nodes = node_attr.to_dict('index')

labeldict = {}
for key, val in dct_nodes.items():
    G.nodes[key]['index'] = val['index']
    G.nodes[key]['size'] = val['size']
    G.nodes[key]['datastate'] = val['datastate']
    labeldict[key] = val['label'] # should change

# create node list for subplot
ids = list(nx.get_node_attributes(G, 'index').values())
vals = list(nx.get_node_attributes(G, 'datastate').values())
nodelst = [id if val == "Yes" else "" for id, val in zip(ids, vals)]
nodelst = list(filter(None, nodelst))

# prepare plot
node_scaling = 2000
size_lst = list(nx.get_node_attributes(G, 'size').values())
size_lst = [x*node_scaling for x in size_lst]

# make plot 
out = os.path.join(outpath, f'MDS_nnodes_{n_nodes}_maxna_{maxna}_ncutoff_{n_cutoff}_perc_{perc}_seed_{seed}.pdf')
fig, axis = plt.subplots(facecolor = 'w', edgecolor = 'k', dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size = size_lst)
nx.draw_networkx_nodes(G, pos, nodelist = nodelst, node_size = 15,
                       node_color = 'black', node_shape = 'x')
plt.savefig(f'{out}')