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

# first not weighted by data state
d_hamming_0 = d_hamming[['node_id', 'p_raw_focal']].drop_duplicates()
d_hamming_0 = d_hamming_0.rename(columns = {'p_raw_focal': 'sum_0'})
d_hamming_0['mean_0'] = d_hamming_0['sum_0']
d_hamming_0['max_0'] = d_hamming_0['sum_0']

# function for getting first n 
def pmass_hamming_above(d_hamming, n_neighbors):
    d_subset = d_hamming[d_hamming['hamming_neighbor'] <= n_neighbors]
    d_pmass = d_subset.groupby('node_id')['p_raw_other'].agg(['sum', 'mean', 'max']).reset_index()
    d_pmass = d_pmass.rename(columns = {'sum': f'sum_{n_neighbors}',
                                        'mean': f'mean_{n_neighbors}',
                                        'max': f'max_{n_neighbors}'})
    #d_pmass['hamming'] = n_neighbors
    return d_pmass 

def pmass_hamming_equal(d_hamming, n_neighbors): 
    d_subset = d_hamming[d_hamming['hamming_neighbor'] == n_neighbors]
    d_pmass = d_subset.groupby('node_id')['p_raw_other'].agg(['sum', 'mean', 'max']).reset_index()
    d_pmass = d_pmass.rename(columns = {'sum': f'sum_{n_neighbors}',
                                        'mean': f'mean_{n_neighbors}',
                                        'max': f'max_{n_neighbors}'})
    #d_pmass['hamming'] = n_neighbors
    return d_pmass

# hamming neighbors acumulating 
n_neighbors = 10
lst_hamming_above = []
lst_hamming_equal = []
for i in range(n_neighbors):
    d_prob_above = pmass_hamming_above(d_hamming, i+1)
    d_prob_equal = pmass_hamming_equal(d_hamming, i+1)
    lst_hamming_above.append(d_prob_above)
    lst_hamming_equal.append(d_prob_equal)

d_hamming_above = reduce(lambda left, right: pd.merge(left, right, on = 'node_id'),
                         lst_hamming_above)
d_hamming_equal = reduce(lambda left, right: pd.merge(left, right, on = 'node_id'),
                         lst_hamming_equal)

# add the first observation
d_hamming_above = d_hamming_above.merge(d_hamming_0, on = 'node_id', how = 'inner')
d_hamming_equal = d_hamming_equal.merge(d_hamming_0, on = 'node_id', how = 'inner')

# FOCUS ON ABOVE FIRST 
comb = list(itertools.combinations(d_hamming_equal['node_id'].values, 2))

# create equal graph
## this is how we SHOULD do it 
Ge = nx.Graph(comb)
dct_nodes = d_hamming_equal.to_dict('index')
for key, val in dct_nodes.items():
    for attr in val: 
        idx = val['node_id']
        Ge.nodes[idx][attr] = val[attr]
        
# create accumulating graph
Ga = nx.Graph(comb)
dct_nodes = d_hamming_above.to_dict('index')
for key, val in dct_nodes.items():
    for attr in val: 
        idx = val['node_id']
        Ga.nodes[idx][attr] = val[attr]

####### MEAN / TRANSPARENCY #########
## problem is that the size becomes so incredibly tiny. 
## so with alpha they completely vanish 
maximum_value = d_hamming_equal['mean_0'].max()
fig, ax = plt.subplots(facecolor = 'w', edgecolor = 'k', dpi = 300)
plt.axis('off')

for i in range(10, -1, -1):
    print(i)

node_size = 2000
for i in range(10,-1,-1): 
    node_size -= 150
    node_alpha = list(nx.get_node_attributes(G, f'mean_{i}').values())
    node_alpha = [x/maximum_value for x in node_alpha]
    nx.draw_networkx_nodes(G, pos, node_size = node_size,
                           alpha = node_alpha, 
                           node_color = '#3182bd')
plt.show();

######## SUM / SIZE ############
d_hamming_equal_long = pd.wide_to_long(d_hamming_equal, 
                                       stubnames = ['sum_', 'mean_', 'max_'], 
                                       i = 'node_id', j = 'val')
maximum_value = d_hamming_equal_long['sum_'].max()

fig, ax = plt.subplots(facecolor = 'w', edgecolor = 'k', dpi = 300)
plt.axis('off')

## size 
### First approach 
from matplotlib import cm
from matplotlib.colors import rgb2hex
n_top = 5
cmap = cm.get_cmap('Oranges', n_top+1)

### accumulating (above) approach
raw_scalar = 10000
for i, j in zip(range(n_top,-1,-1), range(n_top+1)): 
    node_diam = list(nx.get_node_attributes(Ga, f'sum_{i}').values())
    node_diam = [x*raw_scalar for x in node_diam]
    rgba = rgb2hex(cmap(j))
    nx.draw_networkx_nodes(Ga, pos, node_size = node_diam,
                           alpha = 0.8, 
                           node_color = rgba)
plt.show();

### equal approach (not guaranteed that each successive circle is larger)
### actually, this breaks down after NN-5
n_top = 5
cmap = cm.get_cmap('Oranges', n_top+1)
raw_scalar = 20000
for i, j in zip(range(n_top,-1,-1), range(n_top+1)): 
    node_diam = list(nx.get_node_attributes(Ge, f'sum_{i}').values())
    node_diam = [x*raw_scalar for x in node_diam]
    rgba = rgb2hex(cmap(j))
    nx.draw_networkx_nodes(Ge, pos, node_size = node_diam,
                           alpha = 0.8, 
                           node_color = rgba)
plt.show();

### ACCUMULATING W/EDGES 
## this should be fun actually... 
d_id = d_hamming[['p_ind_focal']].drop_duplicates()
d_id = d_id.rename(columns = {'p_ind_focal': 'p_ind_other'})
d_hamming_edges = d_hamming.merge(d_id, on = 'p_ind_other', how = 'inner')
neighbor_dist = d_hamming_edges.groupby(['p_ind_focal', 'p_ind_other'])['hamming_neighbor'].min()
neighbor_dist = neighbor_dist.reset_index(name = 'hamming_distance')
## now add back in the node_ids (from, to) 
d_reference = d_hamming[['node_id', 'p_ind_focal']].drop_duplicates()
neighbor_dist = neighbor_dist.merge(d_reference, on = 'p_ind_focal')
d_reference = d_reference.rename(columns = {'p_ind_focal': 'p_ind_other'})
neighbor_dist = neighbor_dist.merge(d_reference, on = 'p_ind_other')

## create network from this 
Gaw = nx.from_pandas_edgelist(neighbor_dist,
                            'node_id_x',
                            'node_id_y',
                            'hamming_distance')
## add attributes
dct_nodes = d_hamming_above.to_dict('index')
for key, val in dct_nodes.items():
    for attr in val: 
        idx = val['node_id']
        Gaw.nodes[key][attr] = val[attr]
        
# labels
labeldict = {}
for i in Gaw.nodes(): 
    print(i)
    labeldict[i] = i

n_top = 4
cmap = cm.get_cmap('Oranges', n_top+1)
raw_scalar = 5000
node_diams = []
for i, j in zip(range(n_top,-1,-1), range(n_top+1)): 
    node_diam = list(nx.get_node_attributes(Gaw, f'sum_{i}').values())
    node_diam = [x*raw_scalar for x in node_diam]
    rgba = rgb2hex(cmap(j))
    #edge_size = [x if x <= i else 0 for x in weight_lst]
    nx.draw_networkx_nodes(Gaw, pos, node_size = node_diam,
                           alpha = 0.8, 
                           node_color = rgba)
    node_diams.append(node_diam)
    
weight_lst = list(nx.get_edge_attributes(Gaw, 'hamming_distance').values())
for i, j in zip(range(n_top, -1, -1), range(n_top+1)): 
    rgba = rgb2hex(cmap(j))
    node_diam = node_diams[j]
    edge_size = [(1/x)*10 if x == i+1 else 0 for x in weight_lst]
    nx.draw_networkx_edges(Gaw, pos, node_size = node_diam,
                           alpha = 0.8, width = edge_size,
                           edge_color = rgba)
    
nx.draw_networkx_labels(
    Gaw, pos, 
    labels = labeldict, 
    font_size = 20,
    font_color='black')
plt.show();

#### now sanity checking --- adding labels ----
#### I think it is correct though...
allstates = np.loadtxt(f'../data/analysis/allstates_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
allstates[]
val1, val2 = 1, 4
neighbor_dist[(neighbor_dist['node_id_x'] == val1) & (neighbor_dist['node_id_y'] == val2)]

allstates[1027975]
allstates[1027974]

#### are they distant because that bit-flips is unlikely?
# 1, 2 - VERY CLOSE (1 bit-flip)
# 0, 1 - NOT CLOSE (1 bit-flip)

for i in range(len(base_state)): 
    # find idx and probability of flipped state
    base_state_copy = base_state 
    base_state_copy[i] = flip_bit(base_state[i])
    base_state_copy_idx = idx_row_overlap(allstates, base_state_copy)
    base_state_copy_prob = config_prob[base_state_copy_idx]




























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