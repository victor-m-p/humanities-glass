'''
VMP 2022-01-02: 
The plot now reproduces.
'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import networkx as nx
from fun import *

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

##### below has not been revised yet ######

### what is the probability of changing away ###
entry_config_reference = entry_config_master[['entry_id', 'config_id']].drop_duplicates()
entry_config_reference = entry_config_reference.merge(entry_reference, on = 'entry_id', how = 'inner')

def transition_prob(d_main, hamming_dist): 
    d_hamming = d_main[d_main['hamming'] == 1]
    d_hamming = d_hamming[['idx_neighbor', 'prob_neighbor']]
    d_hamming = d_hamming.assign(prob_norm = lambda x: x['prob_neighbor']/sum(x['prob_neighbor']))
    d_hamming = d_hamming.sort_values('prob_neighbor', ascending = False)
    d_hamming = d_hamming.rename(columns = {'idx_neighbor': 'config_id'})
    return d_hamming    

### what is the bit-string of the Roman Imperial Cult?
def uniq_bitstring(allstates, config_idx, question_ids, type):
    focal_config = allstates[config_idx]
    focal_config[focal_config == -1] = 0
    focal_string = ''.join([str(x) for x in focal_config])
    focal_df = pd.DataFrame([focal_config], columns = question_ids)
    focal_df['config_id'] = config_idx
    focal_df = pd.melt(focal_df, id_vars = 'config_id', value_vars = question_ids, var_name = 'related_q_id')
    focal_df = focal_df.rename(columns = {
        'config_id': f'config_id_{type}',
        'value': f'value_{type}'})
    return focal_string, focal_df 

d_main = get_n_neighbors(1, config_idx, allstates, p)
d_transition_prob = transition_prob(d_main, 1)
d_transition_prob = d_transition_prob.merge(entry_config_reference, on = 'config_id', how = 'left')
d_transition_prob # 

question_ids = question_reference['related_q_id'].to_list() 
bitstr_baptist, bitdf_baptist = uniq_bitstring(allstates, 362370, question_ids, 'other')
bitstr_pauline, bitdf_pauline = uniq_bitstring(allstates, 362372, question_ids, 'other')
bitstr_methodist, bitdf_methodist = uniq_bitstring(allstates, config_idx, question_ids, 'focal')

baptist_neighbors = pd.concat([bitdf_baptist, bitdf_pauline])
baptist_difference = baptist_neighbors.merge(bitdf_methodist, on = 'related_q_id', how = 'inner')
baptist_difference = baptist_difference.assign(difference = lambda x: x['value_focal']-x['value_other'])
baptist_difference = baptist_difference[baptist_difference['difference'] != 0]

pd.set_option('display.max_colwidth', None)
baptist_interpret = baptist_difference.merge(question_reference, on = 'related_q_id', how = 'inner')
baptist_interpret = baptist_interpret.rename(columns = {'config_id_other': 'config_id'})

entry_config_reference_uniq = entry_config_reference[['config_id', 'entry_name']].drop_duplicates()
baptist_interpret = entry_config_reference_uniq.merge(baptist_interpret, on = 'config_id', how = 'inner')

d_transition_uniq = d_transition_prob[['config_id', 'prob_norm']].drop_duplicates() 
baptist_interpret = d_transition_uniq.merge(baptist_interpret, on = 'config_id', how = 'inner')
baptist_interpret

# probability for each of the states
p[bitdf_baptist['config_id_other'].unique()[0]] # 0.0291
p[bitdf_pauline['config_id_other'].unique()[0]] # 0.0196
p[bitdf_methodist['config_id_focal'].unique()[0]] # 0.0575

#### old shit (some of it useful, e.g. the loop) ####
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

d_main = get_n_neighbors(1, config_idx, allstates, p)

## prep 
question_ids = question_reference['related_q_id'].to_list() 
d_weight = d_max_weight[['p_norm', 'p_raw', 'config_id']]

## run for the focal state
string_focal, df_focal = uniq_bitstring(allstates, config_idx, question_ids, 'focal')

df_focal
## get all neighboring config_id for this 
## and get the probability of those configurations

## run for selected neighboring states
p_neighbors = [769926, 1027975, 1032071, 638854, 
          1032070, 765830, 765831]
dflst_neighbors = []
bitlst_neighbors = []
for config_index in p_neighbors:
    bitstring_neighbor, df_neighbor = uniq_bitstring(allstates, config_index, question_ids, 'other')
    dflst_neighbors.append(df_neighbor)
    bitlst_neighbors.append(bitstring_neighbor)
df_neighbors = pd.concat(dflst_neighbors)

## merge together
df_complete = df_neighbors.merge(df_focal, on = 'related_q_id', how = 'inner')
df_complete = df_complete.assign(difference = lambda x: x['value_focal']-x['value_other'])
df_complete = df_complete[df_complete['difference'] == 1]

## add entry information 
entry_ids = [230, 738, 424, 1323, 993, 1248, 470]
entry_ref = entry_config_master[entry_config_master['entry_id'].isin(entry_ids)]
entry_ref



entry_ref = entry_ref[['config_id', 'entry_name']].drop_duplicates()
entry_ref = entry_ref.rename(columns = {'config_id': 'config_id_other'}) 

entry_ref


df_complete = df_complete.merge(entry_ref, on = 'config_id_other', how = 'inner')


## add question information
df_complete = df_complete.merge(question_reference, on = 'related_q_id', how = 'inner')
df_complete

## write information about focal state
df_focal = df_focal.merge(question_reference, on = 'related_q_id', how = 'inner')
df_focal

pd.set_option('display.max_colwidth', None)
d_max_weight[d_max_weight['config_id'] == 769927]
