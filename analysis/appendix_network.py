import matplotlib.pyplot as plt 
from matplotlib.colors import rgb2hex
import networkx as nx 
import numpy as np
import pandas as pd 
from fun import *

# setup
n_rows, n_nan, n_nodes, n_top_states = 455, 5, 20, 150

# load data
p = np.loadtxt(f'../data/analysis/p_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')
nodes_reference = pd.read_csv(f'../data/analysis/nref_nrows_455_maxna_5_nodes_20.csv')

# bin states and get likelihood and index
allstates = bin_states(n_nodes) 
d_ind = top_n_idx(n_top_states, p, 'p_ind', 'p_raw') 
d_ind['node_id'] = d_ind.index # 150

d_overlap = datastate_information(d_likelihood, nodes_reference, d_ind) # 407
d_datastate_weight = datastate_weight(d_overlap) # 129

## labels by node_id
### take the maximum p_norm per node_id
### if there are ties do not break them for now
d_max_weight = maximum_weight(d_overlap, d_datastate_weight)

## labels by node_id 
### break ties randomly for now 
node_attr = merge_node_attributes(d_max_weight, d_ind)
node_attr_dict = node_attr.to_dict('index')

# hamming distance
p_ind = d_ind['p_ind'].tolist()
top_states = allstates[p_ind]
h_distances = hamming_distance(top_states) 
h_distances = hamming_edges(n_top_states, h_distances)
h_distances = h_distances[h_distances['hamming'] == 1]
 
# create network
G = nx.from_pandas_edgelist(h_distances,
                            'node_x',
                            'node_y',
                            'hamming')

pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")

# add all node information
node_attr_dict
for idx, val in node_attr_dict.items(): 
    for attr in val: 
        idx = val['node_id']
        G.nodes[idx][attr] = val[attr]
        
# process 
G_full = edge_strength(G, 'p_raw') # fix
edgelst_full, edgew_full = edge_information(G_full, 'pmass_mult', 'hamming', 30000)
nodelst_full, nodesize_full = node_information(G_full, 'p_raw', 5000)

G_data = edge_strength(G, 'datastate_sum')
edgelst_data, edgew_data = edge_information(G_data, 'pmass_mult', 'hamming', 0.2)
nodelst_data, nodesize_data = node_information(G_data, 'datastate_sum', 15)

# plot 
fig, ax = plt.subplots(1, 2, facecolor = 'w', figsize = (14, 8), dpi = 500)
draw_network(G_full, pos, 'Blues', 0.6, nodelst_full, nodesize_full, 
             nodesize_full, edgelst_full, edgew_full, ax[0], 0.5)
draw_network(G_data, pos, 'Blues', 0.6, nodelst_data, nodesize_data, 
             nodesize_data, edgelst_data, edgew_data, ax[1], 0.5)
plt.savefig('../fig/configurations.pdf')
draw_network()

## we need to do the thing with only the full records, 
## and also weighted by that ...