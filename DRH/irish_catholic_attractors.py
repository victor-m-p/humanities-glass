import networkx as nx             # graph manipulation
import numpy as np                # numerical utilities
import matplotlib.pyplot as plt   # plotting
import pandas as pd               # format results
import re
import os 
np.random.seed(1)              # set seed for reproducibility
import configuration as cn 
from fun import * 
from tqdm import tqdm 
from unidecode import unidecode

# https://www.color-hex.com/color-palette/68785
clrs = [
    '#0097c3', # dark blue
    '#9be3f9', # light blue
    '#ffef55', # yellow
    '#f24f26', # red
] 

clrs_nodeedge = [
    '#006b8a', # dark blue
    '#6da4b5', # light blue
    '#b5a93a', # yellow
    '#a83619' # red
]

entry_maxlikelihood = pd.read_csv('../data/analysis/entry_maxlikelihood.csv')
entry_maxlikelihood = entry_maxlikelihood[['config_id', 'entry_name']]

#entry_maxlikelihood = entry_maxlikelihood.groupby('config_id').sample(n=1, random_state=1)
entry_maxlikelihood['entry_name'] = [re.sub(r"(\(.*\))|(\[.*\])", "", x) for x in entry_maxlikelihood['entry_name']]
entry_maxlikelihood['entry_name'] = [re.sub(r"\/", " ", x) for x in entry_maxlikelihood['entry_name']]
entry_maxlikelihood['entry_name'] = [unidecode(text).strip() for text in entry_maxlikelihood['entry_name']]

config_orig = 1017606
d = pd.read_csv(f'../data/COGSCI23/attractors/t0.5_max5000_idx{config_orig}.csv')
d = d[['config_from', 'config_to', 'probability']].drop_duplicates()

# get probability for each state for vertical axis 
config_from = d['config_from'].unique().tolist()
config_to = d['config_to'].unique().tolist()
config_uniq = list(set(config_from + config_to))

configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
p = configuration_probabilities[config_uniq]
config_probs = pd.DataFrame({
    'config_id': config_uniq,
    'config_prob': p
})

config_probs['log_config_prob'] = [np.log(x) for x in config_probs['config_prob']]

## node size by hamming? ## 
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
n_nodes = 20
configurations = bin_states(n_nodes) 
ConfOrig = cn.Configuration(config_orig,
                            configurations,
                            configuration_probabilities)
hamming_distances = []
for i in config_uniq: 
    ConfOther = cn.Configuration(i,
                                configurations,
                                configuration_probabilities)
    distance = ConfOrig.hamming_distance(ConfOther)
    hamming_distances.append((i, distance))

hamming_distances = pd.DataFrame(hamming_distances,
                                columns = ['config_id', 'hamming'])

# fix node colors
attractors = list(set(config_to) - set(config_from))
unique_config_df = pd.DataFrame({'config_id': config_uniq})
node_attributes = unique_config_df.merge(entry_maxlikelihood, on = 'config_id', how = 'left').fillna("")
node_attributes.sort_values('config_id')

# decide on specific labels to include 
entry_names = [(362246, 'Valentinians'),
               (362374, "Jehovah's Witnesses"), # Pauline Christianity, Churches of Christ, Gaengjeongyudo, Monatism, Branch Davidians, Mennonites, Circumcellions, Christianity Ephesus, Jehovah, Egyptian Salafism
               (493318, ''),
               (493446, 'Qumran Movement'),
               (501638, 'Opus Dei'), # Zealots, Muslim, Mourides, Sino-Muslims, Donatism, Northern Irish Roman Catholics
               (1017606, 'Irish Catholics'),
               (1017734, 'Yiguan Dao'),
               (1025538, 'Sokoto'),
               (1025542, ''),
               (1025798, ''),
               (1025926, 'Cistercians') # ....
               ]

node_attributes = node_attributes[['config_id']].drop_duplicates()
entry_names = pd.DataFrame(entry_names, columns = ['config_id', 'entry_name'])
node_attributes = node_attributes.merge(entry_names, on = 'config_id', how = 'inner')

node_attributes['node_color'] = [clrs[0] if x else clrs[1] for x in node_attributes['entry_name']]
node_attributes['node_color'] = [clrs[2] if x == config_orig else y for x, y in zip(node_attributes['config_id'], node_attributes['node_color'])]
node_attributes['node_color'] = [clrs[3] if x in attractors else y for x, y in zip(node_attributes['config_id'], node_attributes['node_color'])]

node_attributes['nodeedge_color'] = [clrs_nodeedge[0] if x else clrs_nodeedge[1] for x in node_attributes['entry_name']]
node_attributes['nodeedge_color'] = [clrs_nodeedge[2] if x == config_orig else y for x, y in zip(node_attributes['config_id'], node_attributes['nodeedge_color'])]
node_attributes['nodeedge_color'] = [clrs_nodeedge[3] if x in attractors else y for x, y in zip(node_attributes['config_id'], node_attributes['nodeedge_color'])]

# for logging
source = node_attributes[node_attributes['config_id'] == config_orig]['entry_name'].tolist()[0]

# add data
naive_path = pd.read_csv(f'../data/COGSCI23/max_attractor/idx{config_orig}.csv')
naive_path = naive_path[['config_from', 'config_to']]
naive_path['edge_color'] = 'k'
d = d.merge(naive_path, on = ['config_from', 'config_to'], how = 'left').fillna('tab:grey')
d['edge_width'] = [x*1.5 if y == 'k' else x*1.5 for x, y in zip(d['probability'], d['edge_color'])]

G = nx.from_pandas_edgelist(d, 
                            source = 'config_from',
                            target = 'config_to',
                            edge_attr = ['edge_color', 'edge_width'],
                            create_using = nx.DiGraph)

# now we start adding stuff 
node_attr = config_probs.merge(node_attributes, on = 'config_id', how = 'inner')
node_attr = node_attr.merge(hamming_distances, on = 'config_id', how = 'inner')
for _, row in node_attr.iterrows(): 
    config_id = int(row['config_id'])
    G.nodes[config_id]['log_config_prob'] = row['log_config_prob']
    G.nodes[config_id]['node_color'] = row['node_color']
    G.nodes[config_id]['nodeedge_color'] = row['nodeedge_color']
    G.nodes[config_id]['hamming'] = row['hamming']
    G.nodes[config_id]['entry_name'] = row['entry_name']
    

# get node attributes out 
pos = {}
nodelist = []
node_color = []
node_size = []
nodeedge_color = []
annotations = {}
for node, attr in G.nodes(data = True): 
    pos[node] = np.array([np.random.normal(0, 0.1), attr['log_config_prob']]) 
    nodelist.append(node)
    node_color.append(attr['node_color'])
    node_size.append(attr['hamming'])
    nodeedge_color.append(attr['nodeedge_color'])
    annotations[node] = attr['entry_name']

# better positions
pos_nudges = {1017606: (0.2, 0), # Irish Catholics
              493318: (0.25, 0), # empty
              1025798: (0.1, 0), # empty
              1017734: (0.50, 0), # Yiguan Dao
              362246: (-0.25, 0), # Valentinians
              493446: (0.22, 0), # Qumran
              1025542: (-0.25, 0), # empty
              1025926: (0.35, 0), # Cistercians
              362374: (-0.2, 0), # Jehovah
              501638: (0.3, 0), # Donatism
              1025538: (-0.40, 0)} # Sokoto 

for key, val in pos.items(): 
    print(key)
    pos_nudge_key = pos_nudges.get(key)
    val[0] += pos_nudge_key[0]
    val[1] += pos_nudge_key[1]
    pos[key] = val

# get edge attributes out 
edge_dict = {}
for config_from, config_to, attr in G.edges(data = True): 
    col = attr['edge_color']
    size = attr['edge_width']
    edge_dict[(config_from, config_to)] = [size, col]

edge_dict_sorted = {k: v for k, v in sorted(edge_dict.items(), key=lambda item: item[1])}
edgelist_sorted = list(edge_dict_sorted.keys())

edge_color = []
edge_width = []
for x in list(edge_dict_sorted.values()):
    edge_width.append(x[0])
    edge_color.append(x[1])
    
# scale stuff
nodesize_scaled = [(x+1)*220 for x in node_size]
weights = [x*4 for x in edge_width]

# visualize the graph
fig, ax = plt.subplots(dpi = 300, figsize = (4, 4)) # (6, 8)
plt.axis('off')
nx.draw_networkx_nodes(G, 
                    pos = pos, 
                    nodelist = nodelist,
                    node_color = node_color,
                    node_size = nodesize_scaled,
                    edgecolors = nodeedge_color,
                    linewidths = [x+1 for x in node_size])
arrows = nx.draw_networkx_edges(G,
                       pos = pos,
                       edgelist = edgelist_sorted,
                       edge_color = edge_color,
                       node_size = [x*1.1 for x in nodesize_scaled],
                       width = weights,
                       arrowstyle = '-|>')

label_options = {"ec": "k", "fc": "white", "alpha": 0.7}                       
nx.draw_networkx_labels(G, pos, annotations, 
                        font_size = 10,
                        bbox = label_options)
for a, w in zip(arrows, weights):
    a.set_joinstyle('miter')
    a.set_capstyle('butt')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.margins(0.15, 0.05)

plt.savefig(f'../fig/{source}_{config_orig}.pdf',
            bbox_inches = 'tight')

# What bit-flips trace the paths? # 
question_reference = pd.read_csv('../data/analysis/question_reference.csv')

bit_flip_list = []
for index, row in d.iterrows(): 
    idx_from = row['config_from']
    idx_to = row['config_to']
    conf_from = cn.Configuration(idx_from,
                                 configurations,
                                 configuration_probabilities)
    conf_to = cn.Configuration(idx_to,
                               configurations,
                               configuration_probabilities)
    bit_flip = conf_from.diverge(conf_to, 
                                 question_reference)
    bit_flip = bit_flip[['question', idx_to]]
    bit_flip['config_from'] = idx_from 
    bit_flip['config_to'] = idx_to
    bit_flip = bit_flip.rename(columns = {idx_to: 'change'})
    bit_flip_list.append(bit_flip)
bit_flip_df = pd.concat(bit_flip_list)

# make it easier to read
node_attr = node_attr[['config_id', 'entry_name']]
node_attr = node_attr.rename(columns = {'config_id': 'config_from',
                                        'entry_name': 'entry_name_from'})
bit_flip_df = bit_flip_df.merge(node_attr, on = 'config_from', how = 'inner')
node_attr = node_attr.rename(columns = {'config_from': 'config_to',
                                        'entry_name_from': 'entry_name_to'})
bit_flip_df = bit_flip_df.merge(node_attr, on = 'config_to', how = 'inner')