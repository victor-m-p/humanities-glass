'''
VMP 2022-01-02: 
Plot layout has changed slightly. 
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

#### Roman Imperial Cult ####
match_soft(entry_config_master, 'Roman Imp')
config_idx = 769927
n_nearest = 2
n_top_states = 49

# get neighbors
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
G = edge_strength(G, 'config_prob') 
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

match_node(annotations, 2) # Mesopotamia (*)
match_node(annotations, 1) # Ancient Egypt (*)
match_node(annotations, 6) # Old Assyrian
match_node(annotations, 3) # Luguru (**)
match_node(annotations, 4) # Pontifex Maximus 
match_node(annotations, 5) # Achaemenid
match_node(annotations, 7) # Archaic cults (**)
match_node(annotations, 0) # Roman Imperial cult

transl_dict = {
    534: 'Roman',
    230: 'Mesopotamia', # *
    424: 'Achaemenid',
    738: 'Ancient Egyptian', # *
    1323: 'Luguru',
    993: 'Pontifical College',
    1248: 'Old Assyrian',
    470: 'Spartan Cults'
}

pos_annot = {
    0: (-50, 370), # Roman
    1: (-120, 370), # Egypt
    2: (-100, -300), # Meso 
    3: (-50, -400), # Luguru
    4: (-130, -350), # Pontifex
    5: (-100, 400), # Achaemenid
    6: (-90, -350), # Old Assyrian
    7: (-100, 250), # Archaic Spartan
}

d_annot = pd.DataFrame.from_dict(transl_dict, 
                       orient = 'index',
                       columns = ['entry_name'])
d_annot['entry_id_drh'] = d_annot.index
d_annot = d_annot.merge(annotations, on = ['entry_id_drh'], how = 'inner')

d_annot = d_annot.iloc[[0, 1, 2, 4, 5, 6, 7, 8]]

### main plot (mixed) ###
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") # reverse code this
#edgew_threshold = [x if x > 0.1 else 0 for x in edgew_full]
nx.draw_networkx_nodes(G, pos, 
                        nodelist = nodelst_sorted,
                        node_size = [x*10000 for x in nodesize_sorted], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G, pos, alpha = 0.7,
                       width = [x*3 if x > 0.05 else 0 for x in edgew_sorted],
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
                #xycoords = 'figure fraction',
                xytext=[pos_x+xx, pos_y+yy],
                #textcoords = 'figure fraction', 
                arrowprops = dict(arrowstyle="->",
                                  connectionstyle='arc3',
                                  color='black'))
plt.savefig('../fig/seed_RomanImpCult_annotated_mix.pdf')


###### below has not been revised #######


### what is the probability of changing away ###
entry_config_reference = d_likelihood[['entry_id', 'p_ind']].drop_duplicates()
entry_config_reference = entry_config_reference.merge(nodes_reference, on = 'entry_id', how = 'inner')

def transition_prob(d_main, hamming_dist): 
    d_hamming = d_main[d_main['hamming'] == 1]
    d_hamming = d_hamming[['idx_neighbor', 'prob_neighbor']]
    d_hamming = d_hamming.assign(prob_norm = lambda x: x['prob_neighbor']/sum(x['prob_neighbor']))
    d_hamming = d_hamming.sort_values('prob_neighbor', ascending = False)
    d_hamming = d_hamming.rename(columns = {'idx_neighbor': 'p_ind'})
    return d_hamming    

### what is the bit-string of the Roman Imperial Cult?
def uniq_bitstring(allstates, config_idx, question_ids, type):
    focal_config = allstates[config_idx]
    focal_config[focal_config == -1] = 0
    focal_string = ''.join([str(x) for x in focal_config])
    focal_df = pd.DataFrame([focal_config], columns = question_ids)
    focal_df['p_ind'] = config_idx
    focal_df = pd.melt(focal_df, id_vars = 'p_ind', value_vars = question_ids, var_name = 'related_q_id')
    focal_df = focal_df.rename(columns = {
        'p_ind': f'p_ind_{type}',
        'value': f'value_{type}'})
    return focal_string, focal_df 

d_main = get_n_neighbors(1, config_idx, allstates, p)
d_transition_prob = transition_prob(d_main, 1)
d_transition_prob = d_transition_prob.merge(entry_config_reference, on = 'p_ind', how = 'left')
d_transition_prob # 

question_ids = question_reference['related_q_id'].to_list() 
bitstr_meso, bitdf_meso = uniq_bitstring(allstates, 769926, question_ids, 'other')
bitstr_achem, bitdf_achem = uniq_bitstring(allstates, 1032071, question_ids, 'other')
bitstr_roman, bitdf_roman = uniq_bitstring(allstates, config_idx, question_ids, 'focal')

baptist_neighbors = pd.concat([bitdf_meso, bitdf_achem])
baptist_difference = baptist_neighbors.merge(bitdf_roman, on = 'related_q_id', how = 'inner')
baptist_difference = baptist_difference.assign(difference = lambda x: x['value_focal']-x['value_other'])
baptist_difference = baptist_difference[baptist_difference['difference'] != 0]

pd.set_option('display.max_colwidth', None)
baptist_interpret = baptist_difference.merge(question_reference, on = 'related_q_id', how = 'inner')
baptist_interpret = baptist_interpret.rename(columns = {'p_ind_other': 'p_ind'})

entry_config_reference_uniq = entry_config_reference[['p_ind', 'entry_name']].drop_duplicates()
baptist_interpret = entry_config_reference_uniq.merge(baptist_interpret, on = 'p_ind', how = 'inner')

d_transition_uniq = d_transition_prob[['p_ind', 'prob_norm']].drop_duplicates() 
baptist_interpret = d_transition_uniq.merge(baptist_interpret, on = 'p_ind', how = 'inner')
baptist_interpret

# probability for each of the states
p[bitdf_meso['p_ind_other'].unique()[0]] # 0.0291
p[bitdf_achem['p_ind_other'].unique()[0]] # 0.0196
p[bitdf_roman['p_ind_focal'].unique()[0]] # 0.0575

### what is the bit-string of the Roman Imperial Cult?
def uniq_bitstring(allstates, config_idx, question_ids, type):
    focal_config = allstates[config_idx]
    focal_config[focal_config == -1] = 0
    focal_string = ''.join([str(x) for x in focal_config])
    focal_df = pd.DataFrame([focal_config], columns = question_ids)
    focal_df['p_ind'] = config_idx
    focal_df = pd.melt(focal_df, id_vars = 'p_ind', value_vars = question_ids, var_name = 'related_q_id')
    focal_df = focal_df.rename(columns = {
        'p_ind': f'p_ind_{type}',
        'value': f'value_{type}'})
    return focal_string, focal_df 

## prep 
question_ids = question_reference['related_q_id'].to_list() 
d_weight = annotations[['p_norm', 'p_raw', 'p_ind']]

## run for the focal state
string_focal, df_focal = uniq_bitstring(allstates, config_idx, question_ids, 'focal')

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
entry_ref = annotations[annotations['entry_id'].isin(entry_ids)]
entry_ref = entry_ref[['p_ind', 'entry_name']].drop_duplicates()
entry_ref = entry_ref.rename(columns = {'p_ind': 'p_ind_other'}) 
df_complete = df_complete.merge(entry_ref, on = 'p_ind_other', how = 'inner')

## add question information
df_complete = df_complete.merge(question_reference, on = 'related_q_id', how = 'inner')
df_complete

## write information about focal state
df_focal = df_focal.merge(question_reference, on = 'related_q_id', how = 'inner')
df_focal

pd.set_option('display.max_colwidth', None)
annotations[annotations['p_ind'] == 769927]

'''



'''








########## old stuff #########


######## free methodists
spartan_node_id = 18
n_nearest = 2
d_spartan = annotations[annotations['node_id'] == spartan_node_id]
spartan_idx = d_spartan['p_ind'].values[0]
#spartan_idx = 769927 # roman imperial
spartan_main = get_n_neighbors(n_nearest, spartan_idx, allstates, p)

## sample the 150 top ones 
n_top_states = 99
spartan_cutoff = spartan_main.sort_values('prob_neighbor', ascending=False).head(n_top_states)
spartan_neighbor = spartan_cutoff[['idx_neighbor', 'prob_neighbor']]
spartan_neighbor = spartan_neighbor.rename(columns = {'idx_neighbor': 'p_ind',
                                                  'prob_neighbor': 'p_raw'})
spartan_focal = spartan_cutoff[['idx_focal', 'prob_focal']].drop_duplicates()
spartan_focal = spartan_focal.rename(columns = {'idx_focal': 'p_ind',
                                                'prob_focal': 'p_raw'})
sparta_ind = pd.concat([spartan_focal, spartan_neighbor])
sparta_ind = sparta_ind.reset_index(drop=True)
sparta_ind['node_id'] = sparta_ind.index

## now it is just the fucking pipeline again. 
sparta_overlap = datastate_information(d_likelihood, nodes_reference, sparta_ind) # 305
sparta_datastate_weight = datastate_weight(sparta_overlap) # 114
sparta_max_weight = maximum_weight(sparta_overlap, sparta_datastate_weight)
sparta_attr = merge_node_attributes(sparta_max_weight, sparta_ind)

## add hamming distance
spartan_hamming_neighbor = spartan_cutoff[['idx_neighbor', 'hamming']]
spartan_hamming_neighbor = spartan_hamming_neighbor.rename(columns = {'idx_neighbor': 'p_ind'})
spartan_hamming_focal = pd.DataFrame([(spartan_idx, 0)], columns = ['p_ind', 'hamming'])
spartan_hamming = pd.concat([spartan_hamming_focal, spartan_hamming_neighbor])
sparta_attr = sparta_attr.merge(spartan_hamming, on = 'p_ind', how = 'inner')
sparta_attr_dict = sparta_attr.to_dict('index')

# hamming distance
p_ind = sparta_ind['p_ind'].tolist()
top_states = allstates[p_ind]
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
node_attr_dict
for idx, val in sparta_attr_dict.items(): 
    for attr in val: 
        idx = val['node_id']
        G.nodes[idx][attr] = val[attr]
        
# process 
G_full = edge_strength(G, 'p_raw') 
edgelst_full, edgew_full = edge_information(G_full, 'pmass_mult', 'hamming', 30000)
nodelst_full, nodesize_full = node_information(G_full, 'p_raw', 5000)

# color by spartan vs. other
color_lst = []
for node in nodelst_full: 
    hamming_dist = sparta_attr_dict.get(node)['hamming']
    color_lst.append(hamming_dist)
    
######### main plot ###########
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") # reverse code this

## slight manual tweak

## now the plot 
nx.draw_networkx_nodes(G_full, pos_mov, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G_full, pos_mov, alpha = 0.7,
                       width = [x*5 for x in edgew_full],
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
plt.savefig('../fig/seed_FreeMethChurch.pdf')

######### reference plot ##########
labeldict = {}
for node in nodelst_full:
    node_id = G_full.nodes[node]['node_id']
    labeldict[node] = node_id

fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") # reverse code this
nx.draw_networkx_nodes(G_full, pos, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G_full, pos, alpha = 0.7,
                       width = edgew_full,
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
label_options = {"ec": "k", "fc": "white", "alpha": 0.1}
nx.draw_networkx_labels(G_full, pos, font_size = 8, labels = labeldict, bbox = label_options)
plt.savefig('../fig/seed_FreeMethChurch_reference.pdf')

match_node(sparta_max_weight, 0) # Free methodist
match_node(sparta_max_weight, 1) # lots of shit
match_node(sparta_max_weight, 2) # Southern Baptists
match_node(sparta_max_weight, 3) # Messalians
match_node(sparta_max_weight, 4) # Nothing (empty)
match_node(sparta_max_weight, 5) # Sachchai
match_node(sparta_max_weight, 9) # Pauline Christianity (45-60 CE)
match_node(sparta_max_weight, 13) # Nothing (empty)

