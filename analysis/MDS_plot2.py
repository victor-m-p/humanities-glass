import numpy as np 
import pandas as pd 
from fun import bin_states, hamming_distance, top_n_idx
from sklearn.manifold import MDS
import itertools
import networkx as nx 
import matplotlib.pyplot as plt
import os 
from matplotlib.patches import Ellipse
from functools import reduce

# setup 
n_rows, n_nan, n_nodes, seed = 455, 5, 20, 254
outpath = '../fig'

# try for 10 nodes, 2 dimensions
d_hamming = pd.read_csv(f'../data/analysis/hamming_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}_ntop_10.csv')
pos = np.loadtxt(f'../data/analysis/pos_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}_ntop_10_ndim_2_kruskal_0.10405095451874981.txt')
d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')
nodes_reference = pd.read_csv(f'../data/analysis/nref_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')
probabilities = np.loadtxt(f'../data/analysis/p_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')

# lookup
d_uniq_nodes = d_hamming[['node_id', 'p_ind_focal']].drop_duplicates()

# quick plot
## WEIGHTING BY DATA STATE MASS
### outer layer
d_size_1 = d_hamming.merge(d_likelihood, on = ['p_ind'])
d_size_1 = d_size_1.assign(config_x_norm = lambda x: x['config_w']*x['p_norm'])
d_size_1 = d_size_1.groupby('p_ind_focal')['config_x_norm'].sum().reset_index(name = 'size_1')
d_size_1 = d_size_1.merge(d_uniq_nodes, on = 'p_ind_focal', how = 'inner')
d_size_1 = d_size_1[['node_id', 'size_1']]

### inner layer
d_size_0 = d_hamming[['node_id', 'p_ind_focal']].drop_duplicates()
d_size_0 = d_size_0.rename(columns = {'p_ind_focal': 'p_ind'})
d_size_0 = d_size_0.merge(d_likelihood, on = 'p_ind', how = 'inner')
d_size_0 = d_size_0.assign(config_x_norm = lambda x: x['p_raw']*x['p_norm'])
d_size_0 = d_size_0.groupby('node_id')['config_x_norm'].sum().reset_index(name = 'size_0')
d_size_0 = d_size_0[['node_id', 'size_0']].drop_duplicates()

## WEIGHTING BY CONFIGURATION MASS
### inner layer
d_prob_0 = d_hamming[['node_id', 'p_ind_focal']].drop_duplicates()
d_prob_0 = d_prob_0.rename(columns = {'p_ind_focal': 'p_ind'})
d_prob_0 = d_prob_0.merge(d_likelihood, on = 'p_ind', how = 'inner')
d_prob_0 = d_prob_0.rename(columns = {'p_raw': 'prob_0'})
d_prob_0 = d_prob_0[['node_id', 'prob_0']].drop_duplicates()

### outer layer
d_prob_1 = d_hamming.merge(d_likelihood, on = ['p_ind'])
d_prob_1 = d_prob_1[['node_id', 'p_ind', 'p_raw']].drop_duplicates()
d_prob_1 = d_prob_1.groupby('node_id')['p_raw'].sum().reset_index(name = 'prob_1')

## merge
d_node_attr = reduce(lambda left, right: pd.merge(left, right, on = 'node_id'),
                     [d_size_0, d_size_1, d_prob_0, d_prob_1])

## create network
comb = list(itertools.combinations(d_node_attr['node_id'].values, 2))

# create network
G = nx.Graph(comb)

dct_nodes = d_node_attr.to_dict('index')
for key, val in dct_nodes.items():
    G.nodes[key]['node_id'] = val['node_id']
    G.nodes[key]['node_size_0'] = val['size_0']
    G.nodes[key]['node_size_1'] = val['size_1']
    G.nodes[key]['node_prob_0'] = val['prob_0']
    G.nodes[key]['node_prob_1'] = val['prob_1']

# size
## setup
size_scaling = 400

## take out values
node_size_0 = list(nx.get_node_attributes(G, 'node_size_0').values())
node_size_1 = list(nx.get_node_attributes(G, 'node_size_1').values())
node_size_1 = [x+y for x, y in zip(node_size_0, node_size_1)]

## scale 
node_size_0_scaled = [x*size_scaling for x in node_size_0]
node_size_1_scaled = [x*size_scaling for x in node_size_1]

#### actually plotting ####
fig, ax = plt.subplots(facecolor = 'w', edgecolor = 'k', dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size = node_size_1_scaled, 
                       #linewidths = nodeedge_size_lst,
                       #edgecolors = 'darkblue',
                       node_color = '#9ecae1')
nx.draw_networkx_nodes(G, pos, node_size = node_size_0_scaled,
                       node_color = '#3182bd')
plt.show();

# weight
size_scaling = 2000

## take out values
node_prob_0 = list(nx.get_node_attributes(G, 'node_prob_0').values())
node_prob_1 = list(nx.get_node_attributes(G, 'node_prob_1').values())
node_prob_1 = [x+y for x, y in zip(node_prob_0, node_prob_1)]

## scale 
node_prob_0_scaled = [x*size_scaling for x in node_prob_0]
node_prob_1_scaled = [x*size_scaling for x in node_prob_1]

#### actually plotting ####
fig, ax = plt.subplots(facecolor = 'w', edgecolor = 'k', dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size = node_prob_1_scaled, 
                       #linewidths = nodeedge_size_lst,
                       #edgecolors = 'darkblue',
                       node_color = '#9ecae1')
nx.draw_networkx_nodes(G, pos, node_size = node_prob_0_scaled,
                       node_color = '#3182bd')
plt.show();

# annotate 
for entry_name, position, node_id in positions: 
    pos_x, pos_y = position 
    abb = transl.get(node_id)
    x_nudge, y_nudge = nudge.get(node_id)
    ax.annotate(abb, xy=[pos_x, pos_y], 
                xytext=[pos_x+x_nudge, pos_y+y_nudge],
                arrowprops = dict(arrowstyle="->",
                                    connectionstyle="arc3"))

plt.savefig(out)