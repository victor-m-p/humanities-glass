import matplotlib.pyplot as plt 
from matplotlib.colors import rgb2hex
import networkx as nx 
import numpy as np
import pandas as pd 
from fun import bin_states, top_n_idx, hamming_distance

def draw_network(Graph, pos, cmap_name, alpha, nodelst, nodesize, nodecolor, edgelst, edgesize, ax_idx, cmap_edge = 1): 
    cmap = plt.cm.get_cmap(cmap_name)
    nx.draw_networkx_nodes(Graph, pos, 
                           nodelist = nodelst,
                           node_size = nodesize, 
                           node_color = nodecolor,
                           linewidths = 0.5, edgecolors = 'black',
                           cmap = cmap,
                           ax = ax[ax_idx])
    rgba = rgb2hex(cmap(cmap_edge))
    nx.draw_networkx_edges(Graph, pos, width = edgesize, 
                        alpha = alpha, edgelist = edgelst,
                        edge_color = rgba,
                        #edge_color = edgesize,
                        #edge_cmap = cmap,
                        ax = ax[ax_idx])
    ax[ax_idx].set_axis_off()

def edge_information(Graph, weight_attribute, filter_attribute, scaling): 
    ## get edge attributes
    edge_weight = nx.get_edge_attributes(Graph, weight_attribute)
    edge_hdist = dict(nx.get_edge_attributes(Graph, filter_attribute))

    ## sorting
    edgew_sorted = {k: v for k, v in sorted(edge_weight.items(), key=lambda item: item[1])}
    edgelst_sorted = list(edgew_sorted.keys())
    edgeh_sorted = dict(sorted(edge_hdist.items(), key = lambda pair: edgelst_sorted.index(pair[0])))
    
    # now we can make lists = edge_w
    edgew_lst = list(edgew_sorted.values())
    edgeh_lst = list(edgeh_sorted.values())

    # now we can filter out elements 
    edgew_threshold = [x if y == 1 else 0 for x, y in zip(edgew_lst, edgeh_lst)]
    edgew_scaled = [x*scaling for x in edgew_threshold]
    return edgelst_sorted, edgew_scaled 

def node_information(Graph, weight_attribute, scaling): 
    # sort nodes 
    node_size = nx.get_node_attributes(Graph, weight_attribute)
    node_sort_size = {k: v for k, v in sorted(node_size.items(), key = lambda item: item[1])}
    nodelst_sorted = list(node_sort_size.keys())
    nodesize_sorted = list(node_sort_size.values())
    nodesize_scaled = [x*scaling for x in nodesize_sorted]
    return nodelst_sorted, nodesize_scaled 

def hamming_edges(n_top_states, h_distances):
    idx = [f'hamming{x}' for x in range(n_top_states)]
    d = pd.DataFrame(h_distances, columns = idx)
    d['node_x'] = d.index
    d = pd.wide_to_long(d, stubnames = "hamming", i = 'node_x', j = 'node_y').reset_index()
    d = d[d['node_x'] != d['node_y']] # remove diagonal 
    d = d.drop_duplicates() # remove duplicates
    return d 

# assign weight information to G 
def edge_strength(G, nodestrength): 
    Gcopy = G.copy()
    for edge_x, edge_y in Gcopy.edges():
        pmass_x = Gcopy.nodes[edge_x][nodestrength]
        pmass_y = Gcopy.nodes[edge_y][nodestrength]
        pmass_mult = pmass_x*pmass_y 
        pmass_add = pmass_x+pmass_y
        Gcopy.edges[(edge_x, edge_y)]['pmass_mult'] = pmass_mult 
        Gcopy.edges[(edge_x, edge_y)]['pmass_add'] = pmass_add  
    return Gcopy 

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

## add likelihood information for the states that appear in top states
def datastate_information(d_likelihood, nodes_reference, d_ind): 
    # merge with nodes reference to get entry_name
    d_likelihood = d_likelihood[['entry_id', 'p_ind', 'p_norm']]
    d_likelihood = d_likelihood.merge(nodes_reference, on = 'entry_id', how = 'inner')
    # make sure that dtypes are preserved 
    d_ind = d_ind.convert_dtypes()
    d_likelihood = d_likelihood.convert_dtypes()
    # merge with d_ind to get data-state probability 
    d_likelihood = d_likelihood.merge(d_ind, on = 'p_ind', indicator = True)
    d_likelihood.rename(columns = {'_merge': 'state'}, inplace = True)
    d_likelihood = d_likelihood.replace({'state': {'left_only': 'only_data', 
                                        'right_only': 'only_config',
                                        'both': 'overlap'}})
    # only interested in states both in data and in top configurations
    d_overlap = d_likelihood[d_likelihood['state'] == 'overlap'].drop(columns={'state'})
    # add information about maximum likelihood 
    max_likelihood = d_overlap.groupby('entry_id')['p_norm'].max().reset_index(name = 'p_norm')
    d_overlap = d_overlap.merge(max_likelihood, on = ['entry_id', 'p_norm'], how = 'left', indicator=True)
    d_overlap = d_overlap.rename(columns = {'_merge': 'max_likelihood'})
    d_overlap = d_overlap.replace({'max_likelihood': {'both': 'yes', 'left_only': 'no'}})
    d_overlap['full_record'] = np.where(d_overlap['p_norm'] == 1, 'yes', 'no')
    return d_overlap 

d_overlap = datastate_information(d_likelihood, nodes_reference, d_ind) # 407

## weight for configurations (proportional to data state weight) 
def datastate_weight(d_overlap): 
    d_entry_node = d_overlap[['entry_id', 'node_id']]
    d_datastate_weight = d_entry_node.groupby('entry_id').size().reset_index(name = 'entry_count')
    d_datastate_weight = d_datastate_weight.assign(entry_weight = lambda x: 1/x['entry_count'])
    d_datastate_weight = d_entry_node.merge(d_datastate_weight, on = 'entry_id', how = 'inner')
    d_datastate_weight = d_datastate_weight.groupby('node_id')['entry_weight'].sum().reset_index(name = 'datastate_sum')
    return d_datastate_weight 

d_datastate_weight = datastate_weight(d_overlap) # 129

## labels by node_id
### take the maximum p_norm per node_id
### if there are ties do not break them for now
def maximum_weight(d_overlap, d_datastate_weight): 
    d_max_weight = d_overlap.groupby('node_id')['p_norm'].max().reset_index(name = 'p_norm')
    d_max_weight = d_overlap.merge(d_max_weight, on = ['node_id', 'p_norm'], how = 'inner')
    d_max_weight = d_datastate_weight.merge(d_max_weight, on = 'node_id', how = 'inner')
    return d_max_weight

d_max_weight = maximum_weight(d_overlap, d_datastate_weight)

## labels by node_id 
### break ties randomly for now 
def merge_node_attributes(d_max_weight, d_ind): 
    d_datastate_attr = d_max_weight.groupby('node_id').sample(n=1, random_state=421)
    d_datastate_attr = d_datastate_attr.drop(columns = {'p_raw'})
    node_attr = d_ind.merge(d_datastate_attr, on = ['node_id', 'p_ind'], how = 'left', indicator = True)
    node_attr = node_attr.rename(columns = {'_merge': 'datastate'})
    node_attr = node_attr.replace({'datastate': {'both': 'yes', 'left_only': 'no'}})
    # configs that are not datastates, fill na (easier later)
    node_attr['datastate_sum'] = node_attr['datastate_sum'].fillna(0)
    node_attr['max_likelihood'] = node_attr['max_likelihood'].fillna('no')
    #node_attr_dict = node_attr.to_dict('index')
    return node_attr

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
G_full = edge_strength(G, 'p_raw') # fix
edgelst_full, edgew_full = edge_information(G_full, 'pmass_mult', 'hamming', 30000)
nodelst_full, nodesize_full = node_information(G_full, 'p_raw', 5000)

G_data = edge_strength(G, 'datastate_sum')
edgelst_data, edgew_data = edge_information(G_data, 'pmass_mult', 'hamming', 0.2)
nodelst_data, nodesize_data = node_information(G_data, 'datastate_sum', 15)

'''
# plot 
fig, ax = plt.subplots(1, 2, facecolor = 'w', figsize = (14, 8), dpi = 500)
draw_network(G_full, pos, 'Blues', 0.6, nodelst_full, nodesize_full, nodesize_full, edgelst_full, edgew_full, 0, 1)
draw_network(G_data, pos, 'Blues', 0.6, nodelst_data, nodesize_data, nodesize_data, edgelst_data, edgew_data, 1, 1)
plt.savefig('../fig/configurations.pdf')
'''

# status 
## (1) need to scale "together" somehow (same min and max, or match the mean?) -- both nodes and edges
## (2) need good way of referencing which records we are talking about
## (3) largest differences between the two plots..?
## (4) could try to run community detection as well 
'''
# reference plot 
labeldict = {}
for node in nodelst_data:
    node_id = G.nodes[node]['node_id']
    labeldict[node] = node_id

fig, ax = plt.subplots(1, 2, figsize = (14, 8), dpi = 500)
draw_network(G_full, pos, 'Blues', 0.6, nodelst_full, nodesize_full, edgelst_full, edgew_full, 0, 1)
draw_network(G_data, pos, 'Blues', 0.6, nodelst_data, nodesize_data, edgelst_data, edgew_data, 1, 1)
label_options = {"ec": "k", "fc": "white", "alpha": 0.1}
nx.draw_networkx_labels(G_full, pos, font_size = 8, labels = labeldict, bbox = label_options, ax = ax[0])
nx.draw_networkx_labels(G_data, pos, font_size = 8, labels = labeldict, bbox = label_options, ax = ax[1])
plt.savefig('../fig/reference_temp.pdf')
'''
# what are in these clusters?

##### COMMUNITIES #####
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

'''
## side-by side plots 
### get communities for the other one as well
comm_lst_data = []
for i in nodelst_data: 
    comm_lst_data.append(comm_dct.get(i))

## plot 
fig, ax = plt.subplots(1, 2, facecolor = 'w', figsize = (14, 8), dpi = 500)
draw_network(G_full, pos, 'Accent', 0.6, nodelst_full, [x*2 for x in nodesize_full], 
             comm_lst_full, edgelst_full, [x*1.25 for x in edgew_full], 0, 5)
draw_network(G_data, pos, 'Accent', 0.6, nodelst_data, [x*1.5 for x in nodesize_data], 
             comm_lst_data, edgelst_data, edgew_data, 1, 5)
plt.savefig('../fig/comm_configurations.pdf')
'''

##### spartans and evangelicals #####
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

