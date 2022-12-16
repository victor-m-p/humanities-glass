'''
VMP 2022-12-16:
Visualization of top configurations (n = 150) for figure 4A.
Writes some tables for reference as well. 
'''

# imports 
import pandas as pd 
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 

# read data
network_information = pd.read_csv('../data/analysis/top_configurations_network.csv')
hamming_information = pd.read_csv('../data/analysis/top_configurations_hamming.csv')

#  add more community details to the network_information dataframe
'''
1. total weight of the commuities (used for table).
2. color of community (perhaps not used). 
'''

## add community weight to data 
community_weight = network_information.groupby('community')['config_prob'].sum().reset_index(name = 'community_weight')
network_information = network_information.merge(community_weight, on = 'community', how = 'inner')

## community color (NB: contingent on cmap)
community_color = {
    0: 'Green',
    1: 'Pastel',
    2: 'Blue',
    3: 'Orange',
    4: 'Grey'
}

network_information['comm_color'] =  network_information['community'].apply(lambda x: community_color.get(x))

## community cmap
cmap_dict = {
    0: 0,
    1: 2,
    2: 4,
    3: 6,
    4: 7
}

network_information['comm_color_code'] = network_information['community'].apply(lambda x: cmap_dict.get(x))

## community labels (descending order of total weight)
comm_order = network_information[['community_weight']].drop_duplicates().reset_index(drop=True)
comm_order['comm_label'] = comm_order.index+1
comm_order['comm_label'] = comm_order['comm_label'].apply(lambda x: f'Group {x}')
network_information = network_information.merge(comm_order, on = 'community_weight', how = 'inner')

# prepare annotations
'''
Manually annotate certain points entries in the plot for reference.
This is a manual process (e.g. picking and giving short names). 
*: other maximum likelihood data-states share configuration.
**: this configuration is not maximum likelihood (but still most likely for config.)
'''

## translation dictionary
transl_dict = {
    230: 'Mesopotamia', # *
    1251: 'Tsonga',
    534: 'Roman',
    654: 'Cistercians', # *
    931: 'Jesuits in Britain', # *
    738: 'Ancient Egyptian', # *
    1043: 'Islam in Aceh',
    1311: 'Jehovah', # *
    879: 'Free Methodist', # *
    984: 'Calvinism', # *
    1010: 'Pythagoreanism', # **
    1304: 'Peyote',
    769: 'Wogeo', # **
    1511: 'Sokoto' # **
}

## to dataframe 
annotations = pd.DataFrame.from_dict(transl_dict, 
                       orient = 'index',
                       columns = ['entry_name'])
annotations['entry_id_drh'] = annotations.index

## because this refers to the entry_id_drh import reference file
entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')
annotations = annotations.merge(entry_reference, on = 'entry_id_drh', how = 'inner')

## now merge with the maximum likelihood dataframe, which is needed because --
## some of the selected states are not in the randomly sampled set subset.  
maxlikelihood_datastates = pd.read_csv('../data/analysis/top_configurations_maxlikelihood.csv')
maxlikelihood_datastates = maxlikelihood_datastates[['entry_id', 'config_id', 'node_id']]
annotations = annotations.merge(maxlikelihood_datastates, on = 'entry_id', how = 'inner')

## merge with the network information to get community information
network_information_subset = network_information[['config_id', 'comm_color_code']].drop_duplicates()
annotations = annotations.merge(network_information_subset, on = 'config_id', how = 'inner') 

## remove a specific node because it is Sokoto twice...
annotations = annotations[annotations['node_id'] != 106] 

## now nudge the position of labels 
pos_annot = {
    0: (300, -30), # Cistercians
    1: (500, 0), # Jesuits
    2: (600, -20), # Egypt
    3: (480, -20), # Jehovah
    4: (300, -20), # Islam
    5: (-90, 350), # Tsonga
    9: (-170, 400), # Meso
    13: (350, -20), # Calvinism
    18: (200, -120), # Free Methodist
    27: (-85, 400), # Roman Imperial
    60: (-600, -10), # Pythagoreanism
    78: (-400, -10), # Sokoto
    93: (-300, -10), # Peyote
    148: (-400, -10) # Wogeo
}

# create network
G = nx.from_pandas_edgelist(hamming_information,
                            'node_x',
                            'node_y',
                            'hamming')

## extract position
pos = {}
for idx, row in network_information.iterrows():
    node_id = row['node_id']
    pos_x = row['pos_x']
    pos_y = row['pos_y']
    pos[node_id] = (pos_x, pos_y)

## add node information to the graph 
network_information_dict = network_information.to_dict('index')
for idx, val in network_information_dict.items(): 
    node_id = val['node_id'] # should also be idx but cautious
    for attr in val: 
        G.nodes[node_id][attr] = val[attr]

# process network information
## check up on this (i.e. can we avoid imports here and make it easy?)
from fun import * 
G = edge_strength(G, 'config_prob') 
edgelist_sorted, edgeweight_sorted = edge_information(G, 'pmass_mult', 'hamming', 30000)

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

# get node attributes
nodelist_sorted, nodesize_sorted = node_attributes(G, 'config_prob', 'config_prob')
_, community_sorted = node_attributes(G, 'config_prob', 'comm_color_code') 
nodesize_sorted

# main plot (Figure 4A)
node_scalar = 10000
fig, ax = plt.subplots(figsize = (6, 8), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Accent")
nx.draw_networkx_nodes(G, pos, 
                        nodelist = nodelist_sorted,
                        node_size = [x*node_scalar for x in nodesize_sorted], 
                        node_color = community_sorted,
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(5))
nx.draw_networkx_edges(G, pos, alpha = 0.7,
                       width = edgeweight_sorted,
                       edgelist = edgelist_sorted,
                       edge_color = rgba
                       )
for index, row in annotations.iterrows(): 
    node_idx = row['node_id']
    name = row['entry_name']
    pos_x, pos_y = pos[node_idx]
    xx, yy = pos_annot.get(node_idx)
    color_code = row['comm_color_code']
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


### now fix the latex tables ###




# write to latex table 
node_attr['comm_weight'] = node_attr['comm_weight'].apply(lambda x: round(x*100, 2))
comm_info = node_attr[['comm_label', 'comm_color', 'comm_weight']].drop_duplicates()
comm_info_latex = comm_info.to_latex(index=False)
with open('comm_info.txt', 'w') as f: 
    f.write(comm_info_latex)

############## ................ ################



# for latex (main figure entry_id)
## add color and weight of community 
d_latex_lookup = d_annot[['entry_name_short', 'entry_name', 'p_ind']]
entry_labels = node_attr[['p_ind', 'comm_label']].drop_duplicates()
entry_labels = entry_labels.dropna()
d_latex_lookup = d_latex_lookup.merge(entry_labels, on = 'p_ind', how = 'inner')
d_latex_lookup = d_latex_lookup.sort_values('comm_label')
d_latex_lookup = d_latex_lookup[['comm_label', 'entry_name_short', 'entry_name']]
latex_lookup_string = d_latex_lookup.to_latex(index=False)
with open('entry_reference.txt', 'w') as f: 
    f.write(latex_lookup_string)

######## ANNOTATION PLOT ########


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
    color_code = row['comm_color_code']
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









############## ................. ###############




### clean this up ###
# biggest differences between communities (to latex)
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

### shared across? ###
pd.set_option('display.max_colwidth', None)
bit_df.groupby('related_q')['weighted_avg_focal'].mean().reset_index(name='mean').sort_values('mean')
bit_df.groupby('related_q')['weighted_avg_focal'].mean().reset_index(name='mean').sort_values('mean', ascending=False)

# three most different per community
comm_color = node_attr[['comm_label', 'community', 'comm_color']].drop_duplicates()
bit_df = bit_df.merge(comm_color, on = 'community', how = 'inner')
bit_diff = bit_df.sort_values(['focal_minus_other_abs'], ascending=False).groupby('community').head(3)
bit_diff = bit_diff.sort_values(['comm_label', 'focal_minus_other_abs'], ascending = [True, False])
bit_diff = bit_diff[['comm_label', 'comm_color', 'question', 'weighted_avg_focal', 'weighted_avg_other', 'focal_minus_other']]

# to latex table (sort communities by total weight)
bit_latex_string = bit_diff.to_latex(index=False)
with open('community_differences.txt', 'w') as f: 
    f.write(bit_latex_string)
