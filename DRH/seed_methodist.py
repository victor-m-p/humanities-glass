'''
VMP 2022-01-02: 
The plot now reproduces.
'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import networkx as nx
from fun import *
pd.set_option('display.max_colwidth', None)

# setup
n_rows, n_nan, n_nodes = 455, 5, 20

def match_node(d, n):
    d = d[d['node_id'] == n][['entry_drh', 'entry_id_drh', 'entry_prob']]
    d = d.sort_values('entry_prob', ascending = False)
    print(d.head(10))

def match_soft(d, s):
    d = d[d['entry_drh'].str.contains(s)]
    print(d.head(10))

#node_attr = pd.read_csv('../data/analysis/node_attr.csv') 
p = np.loadtxt(f'../data/analysis/configuration_probabilities.txt')
entry_config_master = pd.read_csv(f'../data/analysis/entry_configuration_master.csv')
entry_reference = pd.read_csv(f'../data/analysis/entry_reference.csv')
question_reference = pd.read_csv('../data/analysis/question_reference.csv')
network_information = pd.read_csv('../data/analysis/network_information_enriched.csv')

# bin states and get likelihood and index
allstates = bin_states(n_nodes) 

# SEED PIPELINE
def get_n_neighbors(n_neighbors, idx_focal, config_allstates, prob_allstates):
    config_focal = config_allstates[idx_focal]
    prob_focal = prob_allstates[idx_focal]
    lst_neighbors = []
    for idx_neighbor, config_neighbor in enumerate(config_allstates): 
        h_dist = np.count_nonzero(config_focal!=config_neighbor)
        if h_dist <= n_neighbors and idx_focal != idx_neighbor: 
            prob_neighbor = prob_allstates[idx_neighbor]
            lst_neighbors.append((idx_focal, prob_focal, idx_neighbor, prob_neighbor, h_dist ))
    df_neighbor = pd.DataFrame(
        lst_neighbors, 
        columns = ['idx_focal', 'prob_focal', 'idx_neighbor', 'prob_neighbor', 'hamming']
    )
    return df_neighbor

# Find configuration, and entry id for free methodist church
match_soft(entry_config_master, 'Methodist')
config_idx = 362368
n_nearest = 2 # up to hamming distance = 2. 
n_top_states = 49 # perhaps this should be 50 just because that is cleaner

# get the neighbors
d_main = get_n_neighbors(n_nearest, config_idx, allstates, p)

## sample the top ones
d_cutoff =  d_main.sort_values('prob_neighbor', ascending=False).head(n_top_states)
d_neighbor = d_cutoff[['idx_neighbor', 'prob_neighbor']]
d_neighbor = d_neighbor.rename(columns = {'idx_neighbor': 'config_id',
                                          'prob_neighbor': 'config_prob'})
d_focal = d_cutoff[['idx_focal', 'prob_focal']].drop_duplicates()
d_focal = d_focal.rename(columns = {'idx_focal': 'config_id',
                                    'prob_focal': 'config_prob'})
d_ind = pd.concat([d_focal, d_neighbor])
d_ind = d_ind.reset_index(drop=True)
d_ind['node_id'] = d_ind.index

## add hamming distance
d_hamming_neighbor = d_cutoff[['idx_neighbor', 'hamming']]
d_hamming_neighbor = d_hamming_neighbor.rename(columns = {'idx_neighbor': 'config_id'})
d_hamming_focal = pd.DataFrame([(config_idx, 0)], columns = ['config_id', 'hamming'])
d_hamming = pd.concat([d_hamming_focal, d_hamming_neighbor])
node_attr = d_ind.merge(d_hamming, on = 'config_id', how = 'inner')
node_attr_dict = node_attr.to_dict('index')

# hamming distance
config_id = d_ind['config_id'].tolist()
top_states = allstates[config_id]
h_distances = hamming_distance(top_states) 
h_distances = hamming_edges(n_top_states+1, h_distances)
h_distances = h_distances[h_distances['hamming'] == 1]

# create network
G = nx.from_pandas_edgelist(h_distances,
                            'node_x',
                            'node_y',
                            'hamming')

pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")

# add all node information
for idx, val in node_attr_dict.items(): 
    for attr in val: 
        idx = val['node_id']
        G.nodes[idx][attr] = val[attr]
        
# process 
G = edge_strength(G, 'config_prob') # we need p_raw assigned to nodes
edgelst_sorted, edgew_sorted = edge_information(G, 'pmass_mult', 'hamming', 30000)

## thing here is that we need to sort the node information similarly
def node_attributes(Graph, sorting_attribute, value_attribute):
    # first sort by some value (here config_prob)
    sorting_attr = nx.get_node_attributes(G, sorting_attribute)
    sorting_attr = {k: v for k, v in sorted(sorting_attr.items(), key = lambda item: item[1])}
    nodelist_sorted = list(sorting_attr.keys())
    # then take out another thing 
    value_attr = nx.get_node_attributes(G, value_attribute)
    value_attr = {k: v for k, v in sorted(value_attr.items(), key = lambda pair: nodelist_sorted.index(pair[0]))}
    value_sorted = list(value_attr.values())
    # return
    return nodelist_sorted, value_sorted

nodelst_sorted, nodesize_sorted = node_attributes(G, 'config_prob', 'config_prob')

# color by dn vs. other
color_lst = []
for node in nodelst_sorted: 
    hamming_dist = node_attr_dict.get(node)['hamming']
    color_lst.append(hamming_dist)

#### annotations #####
entry_config_weight = entry_config_master[['config_id', 'entry_drh', 'entry_id', 'entry_prob']]
annotations = entry_config_weight.merge(d_ind, on = 'config_id', how = 'inner')
annotations = annotations.merge(entry_reference, on = ['entry_id', 'entry_drh'], how = 'inner')

match_node(annotations, 0) # Free Methodist (*)
match_node(annotations, 1) # Jehovah (*)
match_node(annotations, 2) # Southern Baptist
match_node(annotations, 4) # No MAXLIK (or data-state?)
match_node(annotations, 5) # Sachchai
match_node(annotations, 6) # No MAXLIK (or data-state?)
match_node(annotations, 9) # Tunisian
match_node(annotations, 8) # Pauline
match_node(annotations, 3) # Messalians

transl_dict = {
    879: 'Free Methodist',
    1311: 'Jehovah', # *
    1307: 'S. Baptists',
    953: 'Sachchai', # *
    1517: 'Tunisian Women',
    993: 'Pontifical College',
    196: 'Pauline Christianity',
}

pos_annot = {
    0: (-110, 400), # Free Meth
    1: (-65, 200), # Jehova
    2: (-70, -450), # S. Baptist
    5: (-60, -270), # Sachchai
    9: (-100, -550), # Tunisian
    8: (-145, 350) # Pauline
}

d_annot = pd.DataFrame.from_dict(transl_dict, 
                       orient = 'index',
                       columns = ['entry_name'])
d_annot['entry_id_drh'] = d_annot.index
d_annot = d_annot.merge(annotations, on = ['entry_id_drh'], how = 'inner')

d_annot = d_annot.iloc[[0, 1, 2, 3, 5, 8]]

### main plot (mixed) ###
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Oranges") # reverse code this
nx.draw_networkx_nodes(G, pos, 
                        nodelist = nodelst_sorted,
                        node_size = [x*10000 for x in nodesize_sorted], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.8))
nx.draw_networkx_edges(G, pos, alpha = 0.7,
                       width = [x*5 if x > 0.05 else 0 for x in edgew_sorted],
                       edgelist = edgelst_sorted,
                       edge_color = rgba
                       )
for index, row in d_annot.iterrows(): 
    node_idx = row['node_id']
    name = row['entry_name']
    pos_x, pos_y = pos[node_idx]
    xx, yy = pos_annot.get(node_idx)
    color = rgb2hex(cmap(0.99))
    ax.annotate(name, xy = [pos_x, pos_y],
                color = rgba,
                xytext=[pos_x+xx, pos_y+yy],
                arrowprops = dict(arrowstyle="->",
                                  connectionstyle='arc3',
                                  color='black'))
plt.savefig('../fig/seed_FreeMethChurch_annotated_mix.pdf')

# transition probabilities
import configuration as cn 

# prep 
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
question_reference = pd.read_csv('../data/analysis/question_reference.csv')
configurations = bin_states(n_nodes)  

# initialize Roman
Methodist = cn.Configuration(config_idx, 
                             configurations, 
                             configuration_probabilities)

# transition probabilities
transition_probabilities = Methodist.neighbor_probabilities(configurations,
                                                        configuration_probabilities,
                                                        question_reference)

# these are the transition_probabilities
# if we enforce move. 
transition_probabilities['transition_prob'] = transition_probabilities['transition_prob']*100
transition_probabilities
# most likely (least stable traits)
## 15.01% (0.00029): having large-scale rituals required
## 13.34% (0.00026): having special treatment for corpses
## 12.27% (0.00024): not having formal burials

# least likely (most stable traits)
## 0.02%: having adult sacrifice required
## 0.03%: having child sacrifice required
## 0.19%: having castration required 

# which religions do these correspond to?
## most likely 
most_likely = transition_probabilities.head(3)
most_likely = most_likely[['config_id', 'question', 'transition_prob']]
most_likely = entry_config_master.merge(most_likely, on = 'config_id', how = 'inner')
most_likely = most_likely.sort_values('config_prob', ascending = False)
most_likely
## 1. Soutern Baptists
## 2. Circumcellions
## 3. Sachchai

## least likely 
least_likely = transition_probabilities.tail(3)
least_likely = least_likely[['config_id', 'question', 'transition_prob']]
least_likely = entry_config_master.merge(least_likely, on = 'config_id')
least_likely = least_likely.sort_values('config_prob')
least_likely
## 1. does not exist
## 2. does not exist
## 3. does not exist

# How likely compared to self 
methodist_prob = Methodist.p
methodist_prob # 0.0055 
## more probable than all neighbors
