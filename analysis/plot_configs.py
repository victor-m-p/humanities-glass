####### PLOT DATA STATES #######
import numpy as np 
from fun import top_n_idx, hamming_distance, bin_states
import pandas as pd 
import networkx as nx 
import matplotlib.pyplot as plt 

# setup
n_rows, n_nan, n_nodes, n_top_states = 455, 5, 20, 150

# load stuff
p = np.loadtxt(f'../data/analysis/p_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')
allstates = bin_states(n_nodes) # wow, super fast. 
d_ind = top_n_idx(n_top_states, p, 'p_ind', 'p_val') 
d_ind['index'] = d_ind.index

##### add this to the graph #####
p_ind = d_ind['p_ind'].tolist()
top_states = allstates[p_ind]
h_distances = hamming_distance(top_states) 

idx = [f'hamming{x}' for x in range(n_top_states)]
d = pd.DataFrame(h_distances, columns = idx)
d['node_x'] = d.index
d = pd.wide_to_long(d, stubnames = "hamming", i = 'node_x', j = 'node_y').reset_index()
d = d[d['node_x'] != d['node_y']]
d = d.drop_duplicates() 
d = d[d['hamming'] == 1]

## edge information
d_edge_x = d_ind.rename(columns = {'index': 'node_x',
                                'p_val': 'val_x',
                                'p_ind': 'p_ind_x'})
d_edge_x = d_edge_x.merge(d, on = 'node_x', how = 'inner')

d_edge_y = d_ind.rename(columns = {'index': 'node_y',
                                'p_val': 'val_y',
                                'p_ind': 'p_ind_y'})
d_edge_both = d_edge_y.merge(d_edge_x, on = 'node_y', how = 'inner')
d_edge_both = d_edge_both.assign(edge_mult = lambda x: x['val_y']*x['val_x'])
d_edge_both = d_edge_both.assign(edge_add = lambda x: x['val_y']+x['val_x'])
G = nx.from_pandas_edgelist(d_edge_both,
                            'node_x',
                            'node_y',
                            ['hamming', 'edge_mult', 'edge_add'])

node_attr = d_ind.to_dict('index')
for idx, val in node_attr.items(): 
    for attr in val: 
        idx = val['index']
        G.nodes[idx][attr] = val[attr]

#### multiplication ####
# take out weight
node_size = nx.get_node_attributes(G, 'p_val')

# take out edge weight
edge_weight = nx.get_edge_attributes(G, 'edge_mult')
edge_hdist = nx.get_edge_attributes(G, 'hamming')
# simon scaled this by something funny..
# also, consider sorting 

# to lists
node_size_lst = list(node_size.values())
edge_weight_lst = list(edge_weight.values()) 
edge_hdist_lst = list(edge_hdist.values())

# scale lists
node_scaling = 5000
node_size_lst_scaled = [x*node_scaling for x in node_size_lst]

# 
edge_scaling = 15000
edge_weight_lst_threshold = [x if y == 1 else 0 for x, y in zip(edge_weight_lst, edge_hdist_lst)]
edge_weight_lst_scaled = [x*edge_scaling for x in edge_weight_lst_threshold]
edge_weight_lst_scaled
pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")
cmap = plt.cm.Blues
# plot it 
fig, ax = plt.subplots(facecolor = 'w', dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, 
                       node_size = node_size_lst_scaled, 
                       node_color = node_size_lst_scaled, 
                       linewidths = 0.5, edgecolors = 'black',
                       cmap = cmap)
nx.draw_networkx_edges(G, pos, width = edge_weight_lst_scaled, 
                       alpha = 0.8, 
                       edge_color = edge_weight_lst_scaled,
                       edge_cmap = cmap)

### addition ###
# take out weight
node_size = nx.get_node_attributes(G, 'p_val')

# take out edge weight
edge_weight = nx.get_edge_attributes(G, 'edge_add')
edge_hdist = nx.get_edge_attributes(G, 'hamming')
# simon scaled this by something funny..
# also, consider sorting 

# to lists
node_size_lst = list(node_size.values())
edge_weight_lst = list(edge_weight.values()) 
edge_hdist_lst = list(edge_hdist.values())

# scale lists
node_scaling = 5000
node_size_lst_scaled = [x*node_scaling for x in node_size_lst]

# 
edge_scaling = 100
edge_weight_lst_threshold = [x if y == 1 else 0 for x, y in zip(edge_weight_lst, edge_hdist_lst)]
edge_weight_lst_scaled = [x*edge_scaling for x in edge_weight_lst_threshold]

pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")
cmap = plt.cm.Blues
# plot it 
fig, ax = plt.subplots(facecolor = 'w', dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, 
                       node_size = node_size_lst_scaled, 
                       node_color = node_size_lst_scaled, 
                       linewidths = 0.5, edgecolors = 'black',
                       cmap = cmap)
nx.draw_networkx_edges(G, pos, width = edge_weight_lst_scaled, 
                       alpha = 0.8, 
                       edge_color = edge_weight_lst_scaled,
                       edge_cmap = cmap)

# https://github.com/bhargavchippada/forceatlas2

## plotting with graphviz

####### THE SAME, BUT WHERE WE SORT EDGES #######
G = nx.from_pandas_edgelist(d_edge_both,
                            'node_x',
                            'node_y',
                            ['hamming', 'edge_mult', 'edge_add'])

node_attr = d_ind.to_dict('index')
for idx, val in node_attr.items(): 
    for attr in val: 
        idx = val['index']
        G.nodes[idx][attr] = val[attr]


## get edge attributes
edge_weight = nx.get_edge_attributes(G, 'edge_add')
edge_hdist = dict(nx.get_edge_attributes(G, 'hamming'))

## sorting
edgew_sorted = {k: v for k, v in sorted(edge_weight.items(), key=lambda item: item[1])}
edgelst_sorted = list(edgew_sorted.keys())
edgeh_sorted = dict(sorted(edge_hdist.items(), key = lambda pair: edgelst_sorted.index(pair[0])))

# now we can make lists = edge_w
edgew_lst = list(edgew_sorted.values())
edgeh_lst = list(edgeh_sorted.values())

# now we can filter out elements 
edge_scaling = 100
edgew_threshold = [x if y == 1 else 0 for x, y in zip(edgew_lst, edgeh_lst)]
edgew_scaled = [x*edge_scaling for x in edgew_threshold]

pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")
cmap = plt.cm.Blues

# plot it 
fig, ax = plt.subplots(facecolor = 'w', dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, 
                       node_size = node_size_lst_scaled, 
                       node_color = node_size_lst_scaled, 
                       linewidths = 0.5, edgecolors = 'black',
                       cmap = cmap)
nx.draw_networkx_edges(G, pos, width = edgew_scaled, 
                       alpha = 0.8, edgelist = edgelst_sorted,
                       edge_color = edgew_scaled,
                       edge_cmap = cmap)

######## SIDE-BY-SIDE #########
n_cutoff = 150

d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_455_maxna_5_nodes_20.csv')
nodes_reference = pd.read_csv(f'../data/analysis/nref_nrows_455_maxna_5_nodes_20.csv')

d_ind = top_n_idx(n_cutoff, p, 'p_ind', 'config_w') # same n_cutoff as prep_general.py
d_ind['node_id'] = d_ind.index

#### some of the old stuff ####
## add entry_id information to likelihood dataframe
d_likelihood = d_likelihood.merge(nodes_reference, on = 'entry_id', how = 'inner')

## preserve dtypes (integers) through merge
d_ind = d_ind.convert_dtypes()
d_likelihood = d_likelihood.convert_dtypes()

## merge 
d_nodes = d_likelihood.merge(d_ind, on = 'p_ind', how = 'outer', indicator = True)
d_nodes.rename(columns = {'_merge': 'state'}, inplace = True)

## only states that are in both for some types of information
d_conf = d_nodes[d_nodes['state'] == 'both']

### observation weight (added for normalization of weight)
d_obsw = d_conf.groupby('entry_id').size().reset_index(name = 'entry_count')
d_obsw = d_obsw.assign(entry_weight = lambda x: 1/x['entry_count'])
d_conf = d_conf.merge(d_obsw, on = 'entry_id', how = 'inner')

### weight for configurations (as observed in data states)
node_sum = d_conf.groupby('node_id')['entry_weight'].sum().reset_index(name = 'config_sum')

# subgraph? 
nodes_subgraph = node_sum['node_id'].tolist()

H = G.subgraph(nodes_subgraph)
nx.number_connected_components(G)
nx.number_connected_components(H) 

#### add information 
node_subgraph_dct = node_sum.to_dict('index')
for value in node_subgraph_dct.values(): 
    node_index = value['node_id']
    H.nodes[node_index]['node_id'] = node_index # for reference and sanity
    H.nodes[node_index]['data_w'] = value['config_sum'] 
    
### calculating the values for the edges and nodes again ###
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

# the first plot 
edgelst_G, edgew_G = edge_information(G, 'edge_add', 'hamming', 100)
nodelst_G, nodesize_G = node_information(G, 'p_val', 5000)

def draw_network(Graph, pos, cmap, nodelst, nodesize, edgelst, edgesize, ax_idx): 

    nx.draw_networkx_nodes(Graph, pos, 
                        nodelist = nodelst,
                        node_size = nodesize, 
                        node_color = nodesize, 
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap,
                        ax = ax[ax_idx])
    nx.draw_networkx_edges(Graph, pos, width = edgesize, 
                        alpha = 0.8, edgelist = edgelst,
                        edge_color = edgesize,
                        edge_cmap = cmap,
                        ax = ax[ax_idx])
    ax[ax_idx].set_axis_off()

# the second plot 
edgelst_H, edgew_H = edge_information(H, 'edge_add', 'hamming', 100)
nodelst_H, nodesize_H = node_information(H, 'data_w', 20)

#pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")
#cmap = plt.cm.Blues

fig, ax = plt.subplots(facecolor = 'w', dpi = 300)
#plt.axis('off')
nx.draw_networkx_nodes(H, pos, 
                       nodelist = nodelst_H,
                       node_size = nodesize_H, 
                       node_color = nodesize_H, 
                       linewidths = 0.5, edgecolors = 'black',
                       cmap = cmap)
nx.draw_networkx_edges(G, pos, width = edgew_H, 
                       alpha = 0.8, edgelist = edgelst_H,
                       edge_color = edgew_H,
                       edge_cmap = cmap)

#### as subplots
pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")
cmap = plt.cm.Blues
fig, ax = plt.subplots(1, 2, facecolor = 'w', figsize = (10, 10), dpi = 300)
plt.axis('off')
draw_network(G, pos, cmap, nodelst_G, nodesize_G, edgelst_G, edgew_G, 0)
draw_network(H, pos, cmap, nodelst_H, nodesize_H, edgelst_H, edgew_H, 1)
plt.show();

####### clean version with functions ########

from matplotlib.colors import rgb2hex
def draw_network(Graph, pos, cmap_name, alpha, nodelst, nodesize, edgelst, edgesize, ax_idx): 
    cmap = plt.cm.get_cmap(cmap_name)
    nx.draw_networkx_nodes(Graph, pos, 
                           nodelist = nodelst,
                           node_size = nodesize, 
                           node_color = nodesize,
                           linewidths = 0.5, edgecolors = 'black',
                           cmap = cmap,
                           ax = ax[ax_idx])
    cmap = plt.cm.get_cmap(cmap_name, 2)
    rgba = rgb2hex(cmap(1))
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
d_ind = top_n_idx(n_top_states, p, 'p_ind', 'p_val') 
d_ind['node_id'] = d_ind.index

## add likelihood information
d_likelihood = d_likelihood.merge(nodes_reference, on = 'entry_id', how = 'inner')
max_likelihood = d_likelihood.groupby('entry_id')['p_norm'].max().reset_index(name = 'p_norm')
d_likelihood = d_likelihood.merge(max_likelihood, on = ['entry_id', 'p_norm'], how = 'left', indicator=True)
d_likelihood = d_likelihood.rename(columns = {'_merge': 'max_likelihood'})
d_likelihood = d_likelihood.replace({'max_likelihood': {'both': 'yes', 'left_only': 'no'}})

## preserve dtypes (integers) through merge
d_ind = d_ind.convert_dtypes()
d_likelihood = d_likelihood.convert_dtypes()

## merge 
d_nodes = d_likelihood.merge(d_ind, on = 'p_ind', how = 'outer', indicator = True)
d_nodes.rename(columns = {'_merge': 'state'}, inplace = True)
d_nodes = d_nodes.replace({'state': {'left_only': 'only_data', 
                                     'right_only': 'only_config',
                                     'both': 'overlap'}})

## now focused on plot --
## only the nodes in overlap or configurations 
##### d_nodes can be used as a reference ######
d_configurations = d_nodes[(d_nodes['state'] == 'only_conf') | (d_nodes['state'] == 'overlap')]

### observation weight (added for normalization of weight)
d_observation_weight = d_configurations.groupby('entry_id').size().reset_index(name = 'entry_count')
d_observation_weight = d_observation_weight.assign(entry_weight = lambda x: 1/x['entry_count'])
d_configurations = d_configurations.merge(d_observation_weight, on = 'entry_id', how = 'inner')

### weight for configurations (as observed in data states)
node_weighted = d_configurations.groupby('node_id')['entry_weight'].sum().reset_index(name = 'config_sum')
node_maxlikelihood = d_configurations[d_configurations['max_likelihood'] == 'yes']
node_configs = node_weighted.merge(node_maxlikelihood, on = 'node_id', how = 'left', indicator=True)
node_configs = node_configs.rename(columns = {'_merge': 'maximum_likelihood'})
node_configs = node_configs.replace({'maximum_likelihood': {'both': 'yes',
                                                            'left_only': 'no'}})

## create the ultimate node information dictionary
node_uniq = node_configs.groupby('node_id').sample(n=1, random_state = 421)
node_attr = d_ind.merge(node_uniq, on = ['node_id', 'p_ind'], how = 'left', indicator = True)
node_attr = node_attr.rename(columns = {'_merge': 'datastate'})
node_attr = node_attr.replace({'datastate': {'both': 'yes', 'left_only': 'no'}})
node_attr['config_sum'] = node_attr['config_sum'].fillna(0)
node_attr['maximum_likelihood'] = node_attr['maximum_likelihood'].fillna('no')
node_attr_dict = node_attr.to_dict('index')

# add hamming distance 
p_ind = d_ind['p_ind'].tolist()
top_states = allstates[p_ind]
h_distances = hamming_distance(top_states) 
hamming_full = hamming_edges(n_top_states, h_distances)

#### different layouts (position) #####
#### only the one distance pulls them appart properly ####
## one distance 
hamming_one_distance = hamming_full[hamming_full['hamming'] == 1]
hamming_one_distance['weight'] = hamming_one_distance['hamming']

## maximum value scaling
hamming_maxval = hamming_full['hamming'].max()
hamming_max_scaling = hamming_full.assign(weight = lambda x: hamming_maxval-x['hamming'])

## reciprocal scaling
hamming_reciprocal = hamming_full.assign(weight = lambda x: 1/x['hamming'])

## get positions for all of these 
position_lst = []
for df in [hamming_one_distance, hamming_max_scaling, hamming_reciprocal]: 
    G = nx.from_pandas_edgelist(df, 'node_x', 'node_y', 'weight')
    position = nx.nx_agraph.graphviz_layout(G, prog = 'fdp')
    position_lst.append(position)
pos_one_distance = position_lst[0]
pos_max_scaling = position_lst[1]
pos_reciprocal = position_lst[2]
 
# try to create a network G 
G = nx.from_pandas_edgelist(hamming_one_distance,
                            'node_x',
                            'node_y',
                            'weight')

# add all node information
#### node_info is our REFERENCE ####
for idx, val in node_attr_dict.items(): 
    for attr in val: 
        idx = val['node_id']
        G.nodes[idx][attr] = val[attr]

G_full = edge_strength(G, 'p_val')
edgelst_full, edgew_full = edge_information(G_full, 'pmass_mult', 'weight', 30000)
nodelst_full, nodesize_full = node_information(G_full, 'p_val', 5000)

G_data = edge_strength(G, 'config_sum')
edgelst_data, edgew_data = edge_information(G_data, 'pmass_mult', 'weight', 0.2)
nodelst_data, nodesize_data = node_information(G_data, 'config_sum', 15)

pos = pos_one_distance
fig, ax = plt.subplots(1, 2, facecolor = 'w', figsize = (12, 8), dpi = 500)
plt.axis('off')
draw_network(G_full, pos, 'Blues', 0.6, nodelst_full, nodesize_full, edgelst_full, edgew_full, 0)
draw_network(G_data, pos, 'Blues', 0.6, nodelst_data, nodesize_data, edgelst_data, edgew_data, 1)
plt.savefig('../fig/configurations.pdf')

out = os.path.join(outpath, f'MDS_annotated_nnodes_{n_nodes}_maxna_{n_nan}_ncutoff_{n_cutoff}_perc_{perc}_seed_{seed}.pdf')

### actually, we should scale AFTERWARDS ###
### i.e. maximum node should have same size, and maximum edge should have same 
### thickness --- or the total WEIGHT in the plot should be the same 

### NB: might want to select preferentially the "clean" configs
### rather than just maximum likelihood 


# backtrack some of the civs 
## just check a sample here 


## check those with duplicate vales 
maxL_configs = d_configurations[d_configurations['max_likelihood'] == 'yes']
node_ml_duplicates = maxL_configs.groupby('node_id').size().reset_index(name = 'count')
node_ml_duplicates = node_ml_duplicates[node_ml_duplicates['count'] > 1]
node_ml_duplicates = maxL_configs.merge(node_ml_duplicates, on = 'node_id', how = 'inner')
node_ml_duplicates = node_ml_duplicates.sort_values(['node_id', 'entry_weight'], ascending = [True, False])
node_ml_duplicates = node_ml_duplicates[['node_id', 'entry_name', 'entry_weight']]
node_ml_duplicates.iloc[0:10]

#### save the plot, push and then we can explore w/ Simon

# only states with maximum likelihood 
## this does not do a lot 
## should probably just remove this 
G_maxL = G.copy()
for node in G_maxL.nodes():
    maxL = G_maxL.nodes[node]['maximum_likelihood'] 
    if maxL == 'no':
        G_maxL.nodes[node]['config_sum'] = 0
    else: 
        pass 

G_maxL = edge_strength(G_maxL, 'config_sum')
edgelst_maxL, edgew_maxL = edge_information(G_maxL, 'pmass_mult', 'weight', 0.2)
nodelst_maxL, nodesize_maxL = node_information(G_maxL, 'config_sum', 10)

fig, ax = plt.subplots(1, 2, facecolor = 'w', figsize = (10, 10), dpi = 500)
plt.axis('off')
draw_network(G_full, pos, 'Blues', 0.6, nodelst_full, nodesize_full, edgelst_full, edgew_full, 0)
draw_network(G_maxL, pos, 'Blues', 0.6, nodelst_maxL, nodesize_maxL, edgelst_maxL, edgew_maxL, 1)
plt.show();