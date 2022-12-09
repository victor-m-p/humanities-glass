'''
Would be really good to just do data curation here,
i.e. create the node_attr data and max_weight data,
as well as all of the tables, and then leave the plotting
for another document.
'''

import matplotlib.pyplot as plt 
from matplotlib.colors import rgb2hex
import networkx as nx 
import networkx.algorithms.community as nx_comm
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
comm_weight = node_attr.groupby('community')['p_raw'].sum().reset_index(name = 'comm_weight')
node_attr = node_attr.merge(comm_weight, on = 'community', how = 'inner')
node_attr['p_raw'].sum() # 0.4547

#### community color ##### 
community_color = {
    0: 'Green',
    1: 'Pastel',
    2: 'Blue',
    3: 'Orange',
    4: 'Grey'
}

node_attr['comm_color'] =  node_attr['community'].apply(lambda x: community_color.get(x))

#### community labels #####
comm_order = node_attr[['comm_weight']].drop_duplicates().reset_index(drop=True)
comm_order['comm_label'] = comm_order.index+1
comm_order['comm_label'] = comm_order['comm_label'].apply(lambda x: f'Group {x}')
node_attr = node_attr.merge(comm_order, on = 'comm_weight', how = 'inner')
node_attr[['comm_weight', 'comm_label']].drop_duplicates()

#### community reference ####
node_attr['comm_weight'] = node_attr['comm_weight'].apply(lambda x: round(x*100, 2))
comm_info = node_attr[['comm_label', 'comm_color', 'comm_weight']].drop_duplicates()
comm_info_latex = comm_info.to_latex(index=False)
with open('comm_info.txt', 'w') as f: 
    f.write(comm_info_latex)

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
sref.to_csv('../data/analysis/question_reference.csv', index = False)

## TO LATEX (good)
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

### shared across? ###
pd.set_option('display.max_colwidth', None)
bit_df.groupby('related_q')['weighted_avg_focal'].mean().reset_index(name='mean').sort_values('mean')
bit_df.groupby('related_q')['weighted_avg_focal'].mean().reset_index(name='mean').sort_values('mean', ascending=False)


# three most different per community
comm_color = node_attr[['comm_label', 'community', 'comm_color']].drop_duplicates()
bit_df = bit_df.merge(comm_color, on = 'community', how = 'inner')
bit_diff = bit_df.sort_values(['focal_minus_other_abs'], ascending=False).groupby('community').head(10)
bit_diff = bit_diff.sort_values(['comm_label', 'focal_minus_other_abs'], ascending = [True, False])
bit_diff = bit_diff[['comm_label', 'comm_color', 'question', 'weighted_avg_focal', 'weighted_avg_other', 'focal_minus_other']]

# to latex table (sort communities by total weight)
bit_latex_string = bit_diff.to_latex(index=False)
with open('community_differences.txt', 'w') as f: 
    f.write(bit_latex_string)

bit_diffx[bit_diffx['comm_color'] == 'Pastel'].sort_values('weighted_avg_focal')

''' old approach
#### big latex table ####
ref_observed_other = node_attr[['p_ind', 'p_norm', 'max_likelihood', 'full_record', 'comm_label']].drop_duplicates()
ref_observed_other = ref_observed_other.dropna()
ref_observed_entry = d_max_weight[['entry_name', 'p_ind']]
ref_observed_master = ref_observed_other.merge(ref_observed_entry, on = 'p_ind', how = 'inner')
ref_observed_master = ref_observed_master.sort_values(['comm_label', 'p_norm'], ascending = [True, False])
ref_observed_master['p_norm'] = ref_observed_master['p_norm'].apply(lambda x: round(x*100, 2))
ref_observed_master = ref_observed_master[['comm_label', 'entry_name', 'p_norm']]
ref_observed_latex = ref_observed_master.to_latex(index=False)
with open('ref_observed.txt', 'w') as f: 
    f.write(ref_observed_latex)
'''
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

d_annot = pd.DataFrame.from_dict(transl_dict, 
                       orient = 'index',
                       columns = ['entry_name_short'])
d_annot['entry_id'] = d_annot.index
node_annot = d_max_weight[['p_ind', 'entry_id', 'node_id', 'entry_name']].drop_duplicates()
node_annot = node_annot.dropna()
d_annot = d_annot.merge(node_annot, on = 'entry_id', how = 'inner')
d_annot = d_annot.merge(d_community, on = 'node_id', how = 'inner')
d_annot = d_annot[d_annot['node_id'] != 106] # remove specific one 

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

cmap_dict = {
    0: 0,
    1: 2,
    2: 4,
    3: 6,
    4: 7
}

d_annot['comm_color_code'] = d_annot['community'].apply(lambda x: cmap_dict.get(x))
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

# master dataframes 
comm_color_codes = d_annot[['community', 'comm_color_code']].drop_duplicates()
node_attr = node_attr.merge(comm_color_codes, on = 'community', how = 'inner')
d_max_weight.to_csv('../data/analysis/d_max_weight.csv', index = False)
node_attr.to_csv('../data/analysis/node_attr.csv', index = False) 

######## all configurations #########
# group by entry_id and community 
# take the community that has largest weight for the config. 
# Group | Entry name [DRH id] | Weight 
comm_info = node_attr[['p_ind', 'comm_label']].drop_duplicates()
state_info = d_likelihood[['entry_id', 'p_ind', 'p_norm', 'p_raw']].drop_duplicates()
comm_info = comm_info.dropna()
sate_info = state_info.dropna()
big_table = comm_info.merge(state_info, on = 'p_ind', how = 'inner')
by_comm = big_table.groupby(['entry_id', 'comm_label'])['p_norm'].sum().reset_index(name = 'sum_in_comm')
## only take the highest probability state within each community 
uniq = by_comm.sort_values(['entry_id','sum_in_comm'],ascending=False).groupby('entry_id').head(1)
## take these out 
entry_ref = big_table[['entry_id', 'comm_label']].drop_duplicates()
table = uniq.merge(entry_ref, on = ['entry_id', 'comm_label'], how = 'inner')
table['sum_in_comm'] = table['sum_in_comm'].apply(lambda x: round(x*100, 2))
## merge in entry name
reference = pd.read_csv('../data/analysis/nref_nrows_455_maxna_5_nodes_20.csv')
reference = reference[['entry_id', 'entry_name']].drop_duplicates()
table_names = table.merge(reference, on = 'entry_id', how = 'inner')
table_names = table_names.rename(columns = {
    'comm_label': 'Group',
    'entry_name': 'Entry Name',
    'entry_id': 'DRH ID',
    'sum_in_comm': 'Weight'
})
table_names = table_names.sort_values(['Group', 'Weight'], ascending = [True, False])
table_names = table_names[['Group', 'DRH ID', 'Entry Name', 'Weight']]
table_names['Entry Name'] = table_names[['Entry Name']].replace({r'[^\x00-\x7F]+':''}, regex=True)
pd.set_option('display.max_colwidth', None)
ref_observed_latex = table_names.to_latex(index=False, escape = False)
with open('ref_observed.txt', 'w') as f: 
    f.write(ref_observed_latex)

## only have two religions overlapping communities: 
### 609, 691 

## all of the observed religions that are not in the top 150 ##
### load reference
reference = reference.rename(columns = {'entry_id': 'DRH ID'})
### anti join 
not_top150 = reference.merge(table_names, on = 'DRH ID', how = 'left', indicator = True)
not_top150 = not_top150[not_top150['_merge'] == 'left_only']
not_top150 = not_top150[['DRH ID', 'entry_name']]
not_top150 = not_top150.rename(columns = {'entry_name': 'Entry Name'})
not_top150['Entry Name'] = not_top150[['Entry Name']].replace({r'[^\x00-\x7F]+':''}, regex=True)
ref_unobserved_latex = not_top150.to_latex(index=False, escape = False)
with open('ref_unobserved.txt', 'w') as f: 
    f.write(ref_unobserved_latex)

## sanity check
len(not_top150) + len(table_names) # 407 (great). 

### which attributes shared by all? ###
d_likelihood
sref



'''
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
'''