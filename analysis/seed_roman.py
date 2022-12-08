import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import networkx as nx
from fun import *

#### hand-picking approach ####
def get_match(d, n):
    dm = d[d['node_id'] == n][['entry_name', 'entry_id', 'p_ind', 'p_norm']]
    dm = dm.sort_values('p_norm', ascending = False)
    print(dm.head(10))

# setup
n_rows, n_nan, n_nodes = 455, 5, 20

d_max_weight = pd.read_csv('../data/analysis/d_max_weight.csv')
#node_attr = pd.read_csv('../data/analysis/node_attr.csv') 
p = np.loadtxt(f'../data/analysis/p_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')
nodes_reference = pd.read_csv(f'../data/analysis/nref_nrows_455_maxna_5_nodes_20.csv')
question_reference = pd.read_csv('../data/analysis/question_reference.csv')

# bin states and get likelihood and index
allstates = bin_states(n_nodes) 

####### SEED PIPELINE #######
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

## soft search 
### node_id = 18 (Free Methodist Church)
### node_id = 27 (Roman Imperial Cult)

#### Roman Imperial Cult ####
node_idx = 27
n_nearest = 2
n_top_states = 49

d_idx = d_max_weight[d_max_weight['node_id'] == node_idx]
config_idx = d_idx['p_ind'].values[0]
d_main = get_n_neighbors(n_nearest, config_idx, allstates, p)

## sample the top ones
d_cutoff =  d_main.sort_values('prob_neighbor', ascending=False).head(n_top_states)
d_neighbor = d_cutoff[['idx_neighbor', 'prob_neighbor']]
d_neighbor = d_neighbor.rename(columns = {'idx_neighbor': 'p_ind',
                                                  'prob_neighbor': 'p_raw'})
d_focal = d_cutoff[['idx_focal', 'prob_focal']].drop_duplicates()
d_focal = d_focal.rename(columns = {'idx_focal': 'p_ind',
                                                'prob_focal': 'p_raw'})
d_ind = pd.concat([d_focal, d_neighbor])
d_ind = d_ind.reset_index(drop=True)
d_ind['node_id'] = d_ind.index

## now it is just the fucking pipeline again. 
d_overlap = datastate_information(d_likelihood, nodes_reference, d_ind) # 305
d_datastate_weight = datastate_weight(d_overlap) # 114
d_max_weight = maximum_weight(d_overlap, d_datastate_weight)
d_attr = merge_node_attributes(d_max_weight, d_ind)

## add hamming distance
d_hamming_neighbor = d_cutoff[['idx_neighbor', 'hamming']]
d_hamming_neighbor = d_hamming_neighbor.rename(columns = {'idx_neighbor': 'p_ind'})
d_hamming_focal = pd.DataFrame([(config_idx, 0)], columns = ['p_ind', 'hamming'])
d_hamming = pd.concat([d_hamming_focal, d_hamming_neighbor])
node_attr = d_attr.merge(d_hamming, on = 'p_ind', how = 'inner')
node_attr_dict = node_attr.to_dict('index')

# hamming distance
p_ind = d_ind['p_ind'].tolist()
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
for idx, val in node_attr_dict.items(): 
    for attr in val: 
        idx = val['node_id']
        G.nodes[idx][attr] = val[attr]
        
# process 
G_full = edge_strength(G, 'p_raw') 
edgelst_full, edgew_full = edge_information(G_full, 'pmass_mult', 'hamming', 30000)
nodelst_full, nodesize_full = node_information(G_full, 'p_raw', 5000)

# color by dn vs. other
color_lst = []
for node in nodelst_full: 
    hamming_dist = node_attr_dict.get(node)['hamming']
    color_lst.append(hamming_dist)
    
######### main plot ###########
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") # reverse code this

## slight manual tweak
#pos_mov = pos.copy()
#x, y = pos_mov[1]
#pos_mov[1] = (x, y-10)
'''
##### now the plot #####
nx.draw_networkx_nodes(G_full, pos, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G_full, pos, alpha = 0.7,
                       width = [x*3 for x in edgew_full],
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
plt.savefig('../fig/seed_RomanImpCult.pdf')
'''
'''
######### reference plot (tmp) ##########
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
plt.savefig('../fig/seed_RomanImpCult_reference.pdf')
'''

#### annotations #####
get_match(d_max_weight, 2) # Mesopotamia (*)
get_match(d_max_weight, 1) # Ancient Egypt (*)
get_match(d_max_weight, 6) # Achaemenid (**)
get_match(d_max_weight, 3) # Luguru (**)
get_match(d_max_weight, 4) # Pontifex Maximus 
get_match(d_max_weight, 5) # Old Assyrian
get_match(d_max_weight, 7) # Archaic dn cults (**)
get_match(d_max_weight, 0) # Roman Imperial cult

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
    0: (400, 0), # Roman
    1: (200, 0), # Egypt
    2: (-100, -300), # Meso 
    3: (-300, 250), # Luguru
    4: (-130, -250), # Pontifex
    5: (-300, 0), # Old Assyrian
    6: (-90, -250), # Achaemenid
    7: (-105, 250), # Archaic Spartan
}

d_annot = pd.DataFrame.from_dict(transl_dict, 
                       orient = 'index',
                       columns = ['entry_name_short'])
d_annot['entry_id'] = d_annot.index
d_annot = d_annot.merge(d_max_weight, on = 'entry_id', how = 'inner')
d_annot = d_annot[~d_annot['node_id'].isin([19, 21])]

### main plot (colored) ###
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") # reverse code this
#edgew_threshold = [x if x > 0.1 else 0 for x in edgew_full]
nx.draw_networkx_nodes(G_full, pos, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G_full, pos, alpha = 0.7,
                       width = [x*3 for x in edgew_full],
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
for index, row in d_annot.iterrows(): 
    node_idx = row['node_id']
    name = row['entry_name_short']
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
                                  color=rgba))
plt.savefig('../fig/seed_RomanImpCult_annotated_green.pdf')

### main plot (mixed) ###
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") # reverse code this
#edgew_threshold = [x if x > 0.1 else 0 for x in edgew_full]
nx.draw_networkx_nodes(G_full, pos, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G_full, pos, alpha = 0.7,
                       width = [x*3 for x in edgew_full],
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
for index, row in d_annot.iterrows(): 
    node_idx = row['node_id']
    name = row['entry_name_short']
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

### main plot (black) ###
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
                       width = [x*3 for x in edgew_full],
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
for index, row in d_annot.iterrows(): 
    node_idx = row['node_id']
    name = row['entry_name_short']
    pos_x, pos_y = pos[node_idx]
    xx, yy = pos_annot.get(node_idx)
    color = rgb2hex(cmap(0.99))
    ax.annotate(name, xy = [pos_x, pos_y],
                color = 'black',
                #xycoords = 'figure fraction',
                xytext=[pos_x+xx, pos_y+yy],
                #textcoords = 'figure fraction', 
                arrowprops = dict(arrowstyle="->",
                                  connectionstyle='arc3',
                                  color='black'))
plt.savefig('../fig/seed_RomanImpCult_annotated_black.pdf')

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
d_weight = d_max_weight[['p_norm', 'p_raw', 'p_ind']]

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
entry_ref = d_max_weight[d_max_weight['entry_id'].isin(entry_ids)]
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
d_max_weight[d_max_weight['p_ind'] == 769927]

'''



'''








########## old stuff #########


######## free methodists
spartan_node_id = 18
n_nearest = 2
d_spartan = d_max_weight[d_max_weight['node_id'] == spartan_node_id]
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

get_match(sparta_max_weight, 0) # Free methodist
get_match(sparta_max_weight, 1) # lots of shit
get_match(sparta_max_weight, 2) # Southern Baptists
get_match(sparta_max_weight, 3) # Messalians
get_match(sparta_max_weight, 4) # Nothing (empty)
get_match(sparta_max_weight, 5) # Sachchai
get_match(sparta_max_weight, 9) # Pauline Christianity (45-60 CE)
get_match(sparta_max_weight, 13) # Nothing (empty)

