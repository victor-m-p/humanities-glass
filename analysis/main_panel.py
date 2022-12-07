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
G_full = edge_strength(G, 'p_raw') 
edgelst_full, edgew_full = edge_information(G_full, 'pmass_mult', 'hamming', 30000)
nodelst_full, nodesize_full = node_information(G_full, 'p_raw', 5000)

# status 
## (1) need to scale "together" somehow (same min and max, or match the mean?) -- both nodes and edges
## (2) need good way of referencing which records we are talking about
## (3) largest differences between the two plots..?
## (4) could try to run community detection as well 
# what are in these clusters?

##### COMMUNITIES #####
## can probably add directly to graph actually...
import networkx.algorithms.community as nx_comm
louvain_comm = nx_comm.louvain_communities(G_full, weight = 'hamming', resolution = 0.5, seed = 152) # 8 comm.

# add louvain information
counter = 0
comm_dct = {}
for comm in louvain_comm:
    for node in comm: 
        comm_dct[node] = counter  
    counter += 1

comm_lst_full = []
for i in nodelst_full: 
    comm_lst_full.append(comm_dct.get(i))

######### main plot ###########
fig, ax = plt.subplots(figsize = (6, 8), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Accent")
nx.draw_networkx_nodes(G_full, pos, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = comm_lst_full,
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(5))
nx.draw_networkx_edges(G_full, pos, alpha = 0.7,
                       width = edgew_full,
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
plt.savefig('../fig/community_configs.pdf')

##### community weight #####
d_community = pd.DataFrame.from_dict(comm_dct,
                                     orient='index',
                                     columns = ['community'])
d_community['node_id'] = d_community.index
node_attr = node_attr.merge(d_community, on = 'node_id', how = 'inner')
node_attr.groupby('community')['p_raw'].sum()
node_attr['p_raw'].sum() # 0.4547

#### community color ##### 
community_color = {
    0: 'Green',
    1: 'Pastel',
    2: 'Blue',
    3: 'Orange',
    4: 'Grey'
}

node_attr['color'] =  node_attr['community'].apply(lambda x: community_color.get(x))

##### SREF: Questions IDS #######
sref = pd.read_csv('../data/analysis/sref_nrows_455_maxna_5_nodes_20.csv')

sref_questions_dict = {
    4676: 'Official political support',
    4729: 'Scriptures',
    4745: 'Monumental religious architecture',
    4776: 'Spirit-body distinction',
    4780: 'Belief in afterlife',
    4787: 'Reincarnation in this world',
    4794: 'Special treatment for corpses',
    4808: 'Co-sacrifices in tomb/burial',
    4814: 'Grave goods',
    4821: 'Formal burials',
    4827: 'Supernatural beings present',
    4954: 'Supernatural monitoring present',
    4983: 'Supernatural beings punish',
    5127: 'Castration required',
    5132: 'Adult sacrifice required',
    5137: 'Child sacrifice required',
    5142: 'Suicide required',
    5152: 'Small-scale rituals required',
    5154: 'Large-scale rituals required',
    5220: 'Distinct written language'
}

sref['question'] = sref['related_q_id'].apply(lambda x: sref_questions_dict.get(x))

## TO LATEX
sref_latex = sref[['related_q_id', 'question', 'related_q']]
sref_latex_string = sref_latex.to_latex(index=False)
with open('question_overview.txt', 'w') as f: 
    f.write(sref_latex_string)

##### BIT DIFFERENCE #####
question_ids = sref['related_q_id'].to_list() 
bit_lst = []
for comm in range(5): # five communities 
    idx_focal = list(louvain_comm[comm])
    idx_other = [list(ele) for num, ele in enumerate(louvain_comm) if num != comm]
    idx_other = [item for sublist in idx_other for item in sublist]
    bit_focal = avg_bitstring(allstates, node_attr, question_ids, idx_focal, 'node_id', 'p_ind', 'related_q_id', 'p_raw')
    bit_other = avg_bitstring(allstates, node_attr, question_ids, idx_other, 'node_id', 'p_ind', 'related_q_id', 'p_raw')
    bit_focal = bit_focal.rename(columns = {'weighted_avg': f'weighted_avg_focal'})
    bit_other = bit_other.rename(columns = {'weighted_avg': 'weighted_avg_other'})
    bit_diff = bit_focal.merge(bit_other, on = 'related_q_id', how = 'inner')
    bit_diff = bit_diff.assign(focal_minus_other = lambda x: x[f'weighted_avg_focal']-x['weighted_avg_other'])
    bit_diff['focal_minus_other_abs'] = np.abs(bit_diff['focal_minus_other'])
    bit_diff = sref.merge(bit_diff, on = 'related_q_id', how = 'inner')
    bit_diff = bit_diff.sort_values('focal_minus_other_abs', ascending = False)
    bit_diff['community'] = comm
    bit_lst.append(bit_diff)

# concat
bit_df = pd.concat(bit_lst)
# to percent, and round 
bit_df = bit_df.assign(weighted_avg_focal = lambda x: round(x['weighted_avg_focal']*100, 2),
                       weighted_avg_other = lambda x: round(x['weighted_avg_other']*100, 2),
                       focal_minus_other = lambda x: round(x['focal_minus_other']*100, 2),
                       focal_minus_other_abs = lambda x: round(x['focal_minus_other_abs']*100, 2)
                       )

# three most different per community
comm_color = node_attr[['community', 'color']].drop_duplicates()
bit_df = bit_df.merge(comm_color, on = 'community', how = 'inner')
bit_diff = bit_df.sort_values(['focal_minus_other_abs'], ascending=False).groupby('community').head(3)
bit_diff = bit_diff.sort_values(['community', 'focal_minus_other_abs'], ascending = [True, False])
bit_diff = bit_diff[['community', 'color', 'question', 'weighted_avg_focal', 'weighted_avg_other', 'focal_minus_other']]

# to latex table 
bit_latex_string = bit_diff.to_latex(index=False)
with open('community_differences.txt', 'w') as f: 
    f.write(bit_latex_string)

#### top configurations for each community (maximum likelihood) ####
comm_color = node_attr[['community', 'color', 'p_ind']].drop_duplicates()
d_top_conf = d_max_weight.merge(comm_color, on = 'p_ind', how = 'inner')
# get top three nodes for each community
d_top_nodeid = d_top_conf[['node_id', 'p_raw', 'community']].drop_duplicates()
d_top_nodeid = d_top_nodeid.sort_values('p_raw', ascending=False).groupby('community').head(3)
d_top_nodeid = d_top_nodeid[['node_id', 'community']]
# get the data-states associated with this 
d_top_states = d_top_conf.merge(d_top_nodeid, on = ['node_id', 'community'], how = 'inner')
d_top_states.sort_values('node_id', ascending = True)
# for annotation
d_annotations = d_top_states[['entry_id', 'entry_name']].drop_duplicates()

d_three = d_top_conf.sort_values(['p_raw'], ascending = False).groupby('community').head(3)
d_three = d_three.sort_values(['community', 'p_norm'], ascending = [True, False])

## translation dct 
pd.set_option('display.max_colwidth', None)
d_annotations
entry_translate = {
    543: 'Roman Imperial Cult', #y
    871: 'Spiritualism', #y
    1248: 'Old Assyrian', #y
    1511: 'Sokoto', 
    769: 'Wogeo', #y
    1304: 'Peyote', #y
    862: 'Ilm-e-Khshnoom', #y
    1010: 'Pythagoreanism', #y
    884: 'Pentecostalism', #y
    1371: "Twelver Shi'ism", #
    839: 'German Protestantism',
    654: 'Cistercians',
    926: 'Ladakhi Buddhism',
    'Sichuan Esoteric Buddhist Cult': 'Esoteric Buddhist'
}

d_entry_top = node_attr[['entry_id', 'entry_name']].drop_duplicates()
d_entry_top['entry_short'] = d_entry_top['entry_name'].apply(lambda x: entry_translate.get(x))
d_entry_top = d_entry_top.dropna()

######### plot with labels ###########
top_nodes = d_top_nodeid['node_id'].tolist()
labels_dct = {}
for i in nodelst_full:
    if i in top_nodes: 
        labels_dct[i] = i
    else: 
        labels_dct[i] = ''

fig, ax = plt.subplots(figsize = (6, 8), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Accent")
nx.draw_networkx_nodes(G_full, pos, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = comm_lst_full,
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(5))
nx.draw_networkx_edges(G_full, pos, alpha = 0.7,
                       width = edgew_full,
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
nx.draw_networkx_labels(G_full, pos, labels_dct, font_size = 8)
plt.savefig('../fig/community_configs_labels.pdf')

#### hand-picking approach ####
def get_match(d, n):
    dm = d[d['node_id'] == n][['entry_name', 'entry_id', 'p_norm']]
    dm = dm.sort_values('p_norm', ascending = False)
    print(dm.head(10))

### *: Other religions share this configuration
### **: This religion is not a complete record (but is still maximum likelihood)

## green cluster
get_match(d_max_weight, 9) # Mesopotamia*
get_match(d_max_weight, 5) # Tsonga
get_match(d_max_weight, 27) # Roman Imperial 
## grey cluster
get_match(d_max_weight, 0) # Cistercians*
get_match(d_max_weight, 1) # Jesuits*
get_match(d_max_weight, 2) # Ancient Egypt*
get_match(d_max_weight, 4) # Islam in Aceh
## Orange cluster
get_match(d_max_weight, 3) # Jehova's Witnesses*
get_match(d_max_weight, 18) # Free Methodist Church*
get_match(d_max_weight, 13) # Calvinism*
## blue 
get_match(d_max_weight, 60) # Pythagoreanism**
get_match(d_max_weight, 93) # Peyote
## pastel
get_match(d_max_weight, 148) # Wogeo
get_match(d_max_weight, 78) # Sokoto

transl_dict = {
    230: 'Mesopotamia*',
    1251: 'Tsonga',
    534: 'Roman Imperial',
    654: 'Cistercians*',
    931: 'Jesuits in Britain*',
    738: 'Ancient Egyptian*',
    1043: 'Islam in Aceh',
    1311: 'Jehovah*',
    879: 'Free Methodist*',
    984: 'Calvinism*',
    1010: 'Pythagoreanism**',
    1304: 'Peyote',
    769: 'Wogeo**',
    1511: 'Sokoto**'
}

d_annot = pd.DataFrame.from_dict(transl_dict, 
                       orient = 'index',
                       columns = ['entry_name_short'])
d_annot['entry_id'] = d_annot.index
node_annot = d_max_weight[['node_id', 'entry_id', 'entry_name']].drop_duplicates()
node_annot = node_annot.dropna()
d_annot = d_annot.merge(node_annot, on = 'entry_id', how = 'inner')
d_annot = d_annot.merge(d_community, on = 'node_id', how = 'inner')
d_annot = d_annot[d_annot['node_id'] != 106] # remove specific one 
# for latex 

######## ANNOTATION PLOT ########
d_annot.sort_values('node_id')
pos_annot = {
    0: (300, 0), # Cistercians
    1: (500, 0), # Jesuits
    2: (600, 0), # Egypt
    3: (400, 0), # Jehovah
    4: (300, 0), # Islam
    5: (-80, 300), # Tsonga
    9: (-180, 400), # Meso
    13: (300, 0), # Calvinism
    18: (200, -100), # Free Methodist
    27: (-200, 600), # Roman Imperial
    60: (-600, 0), # Pythagoreanism
    78: (-300, 0), # Sokoto
    93: (-300, 0), # Peyote
    148: (-300, 0) # Wogeo
}

cmap_dict = {
    0: 0,
    1: 2,
    2: 4,
    3: 6,
    4: 7
}

d_annot['color'] = d_annot['community'].apply(lambda x: cmap_dict.get(x))
d_annot

rgb2hex(cmap(2))
# 0: GREEN
# 2: PASTEL
# 4: BLUE
# 6: ORANGE
# 7: GREY

fig, ax = plt.subplots(figsize = (6, 8), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Accent")
nx.draw_networkx_nodes(G_full, pos, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = comm_lst_full,
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(5))
nx.draw_networkx_edges(G_full, pos, alpha = 0.7,
                       width = edgew_full,
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
for index, row in d_annot.iterrows(): 
    node_idx = row['node_id']
    name = row['entry_name_short']
    pos_x, pos_y = pos[node_idx]
    xx, yy = pos_annot.get(node_idx)
    color_code = row['color']
    color = rgb2hex(cmap(color_code))
    ax.annotate(name, xy = [pos_x, pos_y],
                color = color,
                #xycoords = 'figure fraction',
                xytext=[pos_x+xx, pos_y+yy],
                #textcoords = 'figure fraction', 
                arrowprops = dict(arrowstyle="->",
                                  connectionstyle='arc3',
                                  color='black'))
plt.savefig('../fig/community_configs_annotation.pdf')

'''
x_lst = []
y_lst = []
for key, val in pos.items():
    x, y = val
    x_lst.append(x)
    y_lst.append(y)
x_max = np.max(x_lst)
y_max = np.max(y_lst)
pos_scaled = {}
for key, val in pos.items(): 
    x, y = val 
    x = x/x_max
    y = y/y_max 
    pos_scaled[key] = (x, y)
'''

####### SEED PIPELINE #######
# 18: Free Methodist Church
# 35: Archaic Spartan cult. 
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

#### sparta ####
#spartan_node_id = 35
n_nearest = 2
#d_spartan = d_max_weight[d_max_weight['node_id'] == spartan_node_id]
#spartan_idx = d_spartan['p_ind'].values[0]
spartan_idx = 769927 # roman imperial
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
pos_mov = pos.copy()
x, y = pos_mov[1]
pos_mov[1] = (x, y-10)

## now the plot 
nx.draw_networkx_nodes(G_full, pos_mov, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G_full, pos_mov, alpha = 0.7,
                       width = [x*3 for x in edgew_full],
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
plt.savefig('../fig/seed_RomanImpCult.pdf')

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
plt.savefig('../fig/seed_RomanImpCult_reference.pdf')

get_match(sparta_max_weight, 2) # Mesopotamia, Thessalians
get_match(sparta_max_weight, 1) # Egypt
get_match(sparta_max_weight, 6) # Achaemenid 
get_match(sparta_max_weight, 3) # Luguru
get_match(sparta_max_weight, 4) # Pontifex Maximus
get_match(sparta_max_weight, 5) # Old Assyrian
get_match(sparta_max_weight, 7) # Archaic Spartan cults

## table for the top states ##
n_labels = 10
sparta_ind_raw = sparta_max_weight[['p_ind', 'p_raw']].drop_duplicates()
sparta_top_configs = sparta_ind_raw.sort_values('p_raw', ascending = False).head(n_labels)
sparta_top_configs = sparta_max_weight.merge(sparta_top_configs, on = ['p_ind', 'p_raw'], how = 'inner')
sparta_max_weight[sparta_max_weight['p_ind'] == spartan_idx]

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

## which traits would have to be changed ##
# ...

##### features ######
## we need to scale this differently: 
## (1) we need to weigh by probability of configuration
## (2) take the weighted mean configuration for each 
## (3) compare against the weighted mean of the rest of the states

sref = pd.read_csv('../data/analysis/sref_nrows_455_maxna_5_nodes_20.csv')
question_ids = sref['related_q_id'].to_list()

def state_agreement(d, config_lst): 
    
    # subset states 
    p_ind_uniq = d[d['node_id'].isin(config_lst)]
    p_ind_uniq = p_ind_uniq['p_ind'].unique()
    p_ind_uniq = list(p_ind_uniq)

    # get the configurations
    d_conf = allstates[p_ind_uniq]

    # to dataframe 
    d_mat = pd.DataFrame(d_conf, columns = question_ids)
    d_mat['p_ind'] = p_ind_uniq
    d_mat = pd.melt(d_mat, id_vars = 'p_ind', value_vars = question_ids, var_name = 'related_q_id')
    d_mat = d_mat.replace({'value': {-1: 0}})
    d_mat = d_mat.groupby('related_q_id')['value'].mean().reset_index(name = 'mean_val')

    # merge back in question names
    d_interpret = d_mat.merge(sref, on = 'related_q_id', how = 'inner')
    d_interpret = d_interpret.sort_values('mean_val')

    # return 
    return d_interpret

# run on the big communities
pd.set_option('display.max_colwidth', None)

def disagreement_across(d):
    d_std = d.groupby('related_q')['mean_val'].std().reset_index(name = 'standard_deviation')
    d_mean = d.groupby('related_q')['mean_val'].mean().reset_index(name = 'mean_across')
    d_final = d_std.merge(d_mean, on = 'related_q', how = 'inner')
    d_final = d_final.sort_values('standard_deviation', ascending=False)
    return d_final 

##### labels #####
# for this we should do the top 3 (or something) most highly
# weighted configurations per community and then take the corresponding
# religions (preferrably unique, maybe only considering unique). 
def get_match(d, n):
    dm = d[d['node_id'] == n][['entry_name']]
    print(dm.head(10))

get_match(18) # Free Methodist Church
get_match(27) # Roman imperial cult
get_match(35) # Archaic Spartan cult. 

########### old shit ###########
def state_agreement(d, config_lst): 
    
    # subset states 
    p_ind_uniq = d[d['node_id'].isin(config_lst)]
    p_ind_uniq = p_ind_uniq['p_ind'].unique()
    p_ind_uniq = list(p_ind_uniq)

    # get the configurations
    d_conf = allstates[p_ind_uniq]

    # to dataframe 
    d_mat = pd.DataFrame(d_conf, columns = question_ids)
    d_mat['p_ind'] = p_ind_uniq
    d_mat = pd.melt(d_mat, id_vars = 'p_ind', value_vars = question_ids, var_name = 'related_q_id')
    d_mat = d_mat.replace({'value': {-1: 0}})
    d_mat = d_mat.groupby('related_q_id')['value'].mean().reset_index(name = 'mean_val')

    # merge back in question names
    d_interpret = d_mat.merge(sref, on = 'related_q_id', how = 'inner')
    d_interpret = d_interpret.sort_values('mean_val')

    # return 
    return d_interpret

# run on the big communities
pd.set_option('display.max_colwidth', None)

def disagreement_across(d):
    d_std = d.groupby('related_q')['mean_val'].std().reset_index(name = 'standard_deviation')
    d_mean = d.groupby('related_q')['mean_val'].mean().reset_index(name = 'mean_across')
    d_final = d_std.merge(d_mean, on = 'related_q', how = 'inner')
    d_final = d_final.sort_values('standard_deviation', ascending=False)
    return d_final 