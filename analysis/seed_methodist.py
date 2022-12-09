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

#### Free Methodist ####
node_idx = 18
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
                        node_size = [x*4 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G_full, pos, alpha = 0.7,
                       width = [x*5 for x in edgew_full],
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
label_options = {"ec": "k", "fc": "white", "alpha": 0.1}
nx.draw_networkx_labels(G_full, pos, font_size = 8, labels = labeldict, bbox = label_options)
plt.savefig('../fig/seed_FreeMethChurch_reference.pdf')

#### annotations #####
get_match(d_max_weight, 0) # Free Methodist (*)
get_match(d_max_weight, 1) # Jehovah (*)
get_match(d_max_weight, 2) # Southern Baptist
get_match(d_max_weight, 4) # No maximum likelihood (or data state?)
get_match(d_max_weight, 5) # Sachchai
get_match(d_max_weight, 6) # Tunisian Women
get_match(d_max_weight, 9) # Pauline
get_match(d_max_weight, 8) # No maximum likelihood (or data state?)

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
    0: (-95, 360), # Free Meth
    1: (-48, -200), # Jehova
    2: (-65, -250), # S. Baptist
    5: (-55, -270), # Sachchai
    6: (-95, -300), # Tunisian
    9: (-122, 280) # Pauline
}

d_annot = pd.DataFrame.from_dict(transl_dict, 
                       orient = 'index',
                       columns = ['entry_name_short'])
d_annot['entry_id'] = d_annot.index
d_annot = d_annot.merge(d_max_weight, on = 'entry_id', how = 'inner')
d_annot = d_annot[~d_annot['node_id'].isin([15, 28])]

### main plot (colored) ###
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Oranges") # reverse code this
nx.draw_networkx_nodes(G_full, pos, 
                        nodelist = nodelst_full,
                        node_size = [x*4 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.8))
nx.draw_networkx_edges(G_full, pos, alpha = 0.7,
                       width = [x*5 for x in edgew_full],
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
plt.savefig('../fig/seed_FreeMethChurch_annotated_orange.pdf')

### main plot (mixed) ###
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
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
plt.savefig('../fig/seed_FreeMethChurch_annotated_mix.pdf')

### main plot (black) ###
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
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
plt.savefig('../fig/seed_FreeMethChurch_annotated_black.pdf')

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
bitstr_baptist, bitdf_baptist = uniq_bitstring(allstates, 362370, question_ids, 'other')
bitstr_pauline, bitdf_pauline = uniq_bitstring(allstates, 362372, question_ids, 'other')
bitstr_methodist, bitdf_methodist = uniq_bitstring(allstates, config_idx, question_ids, 'focal')

baptist_neighbors = pd.concat([bitdf_baptist, bitdf_pauline])
baptist_difference = baptist_neighbors.merge(bitdf_methodist, on = 'related_q_id', how = 'inner')
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
p[bitdf_baptist['p_ind_other'].unique()[0]] # 0.0291
p[bitdf_pauline['p_ind_other'].unique()[0]] # 0.0196
p[bitdf_methodist['p_ind_focal'].unique()[0]] # 0.0575


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
d_weight = d_max_weight[['p_norm', 'p_raw', 'p_ind']]

## run for the focal state
string_focal, df_focal = uniq_bitstring(allstates, config_idx, question_ids, 'focal')

df_focal
## get all neighboring p_ind for this 
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
entry_ref = d_likelihood[d_likelihood['entry_id'].isin(entry_ids)]
entry_ref



entry_ref = entry_ref[['p_ind', 'entry_name']].drop_duplicates()
entry_ref = entry_ref.rename(columns = {'p_ind': 'p_ind_other'}) 

entry_ref


df_complete = df_complete.merge(entry_ref, on = 'p_ind_other', how = 'inner')


## add question information
df_complete = df_complete.merge(question_reference, on = 'related_q_id', how = 'inner')
df_complete

## write information about focal state
df_focal = df_focal.merge(question_reference, on = 'related_q_id', how = 'inner')
df_focal

pd.set_option('display.max_colwidth', None)
d_max_weight[d_max_weight['p_ind'] == 769927]
