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