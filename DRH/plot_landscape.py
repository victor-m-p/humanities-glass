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
annotations = annotations[annotations['node_id'] != 101] 
annotations.sort_values('node_id')

## now nudge the position of labels 
pos_annot = {
    0: (-400, -30), # Cistercians
    1: (400, 0), # Egypt
    2: (450, -20), # Jesuit
    3: (-500, 0), # Jehovah
    4: (-400, -20), # Islam
    9: (300, 0), # Tsonga
    11: (-700, 0), # Calvinism
    12: (250, -20), # Meso
    16: (-600, 0), # Free Methodist
    24: (200, 0), # Roman Imperial
    54: (200, 100), # Pythagoreanism
    79: (-400, -10), # Sokoto
    108: (200, 100), # Peyote
    138: (-400, -10) # Wogeo
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
G = edge_strength(G, 'config_prob') # would be nice to get rid of this. 
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

########## TABLES ##########
# table with all entry_id that appear in a community 
'''
Locate all entry_id that are "in" a community, 
find the community that they have most probability weight in. 
Save to latex table with columns
(group, entry_id_drh, entry_name, weight)
'''

## load the information on top configuration / entry overlap
config_entry_overlap = pd.read_csv('../data/analysis/top_configurations_overlap.csv')
## add community information
network_information_sub = network_information[['config_id', 'comm_label']]
config_entry_overlap = config_entry_overlap.merge(network_information_sub)
## groupby community and entry id 
config_entry_comm = config_entry_overlap.groupby(['comm_label', 'entry_id'])['entry_prob'].sum().reset_index()
## if there is a tie between two communities 
## for a specific entry_id, then take first
config_entry_comm = config_entry_comm.sort_values('entry_prob', ascending=False).groupby('entry_id').head(1)
## add back name and get the DRH id 
entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')
config_entry_comm = config_entry_comm.merge(entry_reference, on = 'entry_id', how = 'inner')
## select columns 
config_entry_comm = config_entry_comm[['comm_label', 'entry_id_drh', 'entry_drh', 'entry_prob']]
## sort 
config_entry_comm = config_entry_comm.sort_values(['comm_label', 'entry_prob', 'entry_id_drh'], ascending = [True, False, True])
## rename columns 
config_entry_comm = config_entry_comm.rename(
    columns = {'comm_label': 'Group',
               'entry_id_drh': 'DRH ID',
               'entry_drh': 'Entry name (DRH)',
               'entry_prob': 'Weight'})
## to latex and save
config_entry_latex = config_entry_comm.to_latex(index=False)
with open('../tables/top_config_included.txt', 'w') as f: 
    f.write(config_entry_latex)

# table with all entries (entry_id) that do not appear in top states
'''
Take all of the entries in our data that do not appear in any community
(i.e. who only have configurations that are not in the top n = 150 configurations).
Save to latex table with columns (entry_id_drh, entry_name)

'''
## anti-join 
top_config_entries = config_entry_overlap[['entry_id']]
excluded_entries = entry_reference.merge(top_config_entries, on = 'entry_id', how = 'left', indicator = True)
excluded_entries = excluded_entries[excluded_entries['_merge'] == 'left_only']
## select columns
excluded_entries = excluded_entries[['entry_id_drh', 'entry_drh']]
## sort values 
excluded_entries = excluded_entries.sort_values('entry_id_drh', ascending = True)
## rename columns 
excluded_entries = excluded_entries.rename(
    columns = {'entry_id_drh': 'DRH ID',
               'entry_drh': 'Entry name (DRH)'})
## to latex and save
excluded_entries_latex = excluded_entries.to_latex(index=False)
with open('../tables/top_config_excluded.txt', 'w') as f: 
    f.write(excluded_entries_latex)

# table with distinctive features for each community
'''
We should make this cleaner.
'''

## read question reference
question_reference = pd.read_csv('../data/analysis/question_reference.csv')
## get the list of questions
question_id_list = question_reference['question_id'].tolist()
## generate allstates 
n_nodes = 20 
allstates = bin_states(n_nodes)
## loop through five communities
## consider rewriting this (this can be made better)
bit_lst = []
for comm in range(5): # five communities 
    idx_focal = network_information[network_information['community'] == comm]['node_id'].tolist()
    idx_other = network_information[network_information['community'] != comm]['node_id'].tolist()
    bit_focal = avg_bitstring(allstates, network_information, question_id_list, idx_focal, 'node_id', 'config_id', 'question_id', 'config_prob')
    bit_other = avg_bitstring(allstates, network_information, question_id_list, idx_other, 'node_id', 'config_id', 'question_id', 'config_prob')
    bit_focal = bit_focal.rename(columns = {'weighted_avg': f'weighted_avg_focal'})
    bit_other = bit_other.rename(columns = {'weighted_avg': 'weighted_avg_other'})
    bit_diff = bit_focal.merge(bit_other, on = 'question_id', how = 'inner')
    bit_diff = bit_diff.assign(focal_minus_other = lambda x: x[f'weighted_avg_focal']-x['weighted_avg_other'])
    bit_diff['focal_minus_other_abs'] = np.abs(bit_diff['focal_minus_other'])
    bit_diff = question_reference.merge(bit_diff, on = 'question_id', how = 'inner')
    bit_diff = bit_diff.sort_values('focal_minus_other_abs', ascending = False)
    bit_diff['community'] = comm
    bit_lst.append(bit_diff)

# concat
bit_df = pd.concat(bit_lst)

# to percent, and round 
bit_df = bit_df.assign(weighted_avg_focal = lambda x: round(x['weighted_avg_focal']*100, 2),
                       weighted_avg_other = lambda x: round(x['weighted_avg_other']*100, 2),
                       focal_minus_other = lambda x: round(x['focal_minus_other']*100, 2)
                       )

# three most different per community
comm_color = network_information[['comm_label', 'community', 'comm_color']].drop_duplicates()
bit_df = bit_df.merge(comm_color, on = 'community', how = 'inner')
# top three most distinctive features
bit_diff = bit_df.sort_values(['focal_minus_other_abs'], ascending=False).groupby('community').head(3)
# sort values
bit_diff = bit_diff.sort_values(['comm_label', 'focal_minus_other_abs'], ascending = [True, False])
# select columns
bit_diff = bit_diff[['comm_label', 'comm_color', 'question', 'weighted_avg_focal', 'weighted_avg_other', 'focal_minus_other']]
# rename columns
bit_diff = bit_diff.rename(columns = {'comm_label': 'Group',
                                      'comm_color': 'Color',
                                      'question': 'Question',
                                      'weighted_avg_focal': 'Avg. S',
                                      'weighted_avg_other': 'Avg. O',
                                      'focal_minus_other': 'Diff'
                                      })
# to latex table 
bit_latex_string = bit_diff.to_latex(index=False)
with open('../tables/community_questions_table.txt', 'w') as f: 
    f.write(bit_latex_string)

# table with information on the entries that we highlight/annotate in 4A
'''
Save to latex table with columns: 
(community, entry_name, entry_name_drh)
'''
## get the community labels (groups)
network_information_comm = network_information[['comm_label', 'config_id']]
## merge with annotation dataframe
annotation_table = network_information_comm.merge(annotations, on = 'config_id', how = 'inner')
## select subset of columns
annotation_table = annotation_table[['comm_label', 'entry_name', 'entry_drh']]
## rename columns 
annotation_table = annotation_table.rename(
    columns = {'comm_label': 'Group',
               'entry_name': 'Entry name (short)',
               'entry_drh': 'Entry name (DRH)'})
## to latex 
annotation_latex = annotation_table.style.hide(axis = 'index').to_latex()
## save 
with open('../tables/annotation_table.txt', 'w') as f: 
    f.write(annotation_latex)

# table with total probability mass per community
'''
Not included in the manuscript, but used as reference
'''
## select needed columns
community_table = network_information[['comm_label', 'community_weight']].drop_duplicates()
## convert from fraction to percentage and round to 2 decimals
community_table['community_weight'] = community_table['community_weight'].apply(lambda x: round(x*100, 2))
## rename columns
community_table = community_table.rename(
    columns = {'comm_label': 'Group',
               'community_weight': 'Weight'}
)
## to latex table (might want to come back and fix decimals)
community_table = community_table.to_latex(index = False)
## save 
with open('../tables/community_weight_table.txt', 'w') as f: 
    f.write(community_table)

# save additional information
network_information.to_csv('../data/analysis/network_information_enriched.csv', index = False)