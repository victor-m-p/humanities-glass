'''
VMP 2022-12-12: 
Visualize the paratmers (Jij, hi) versus surface-level correlations. 
Produces figure 3A, 3B. 
'''

import itertools 
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 
import numpy as np
from fun import *

def node_edge_lst(n, corr_J, means_h): 
    nodes = [node+1 for node in range(n)]
    comb = list(itertools.combinations(nodes, 2))
    d_edgelst = pd.DataFrame(comb, columns = {'n1', 'n2'})
    d_edgelst['weight'] = corr_J
    d_nodes = pd.DataFrame(nodes, columns = {'n'})
    d_nodes['size'] = means_h
    d_nodes = d_nodes.set_index('n')
    dct_nodes = d_nodes.to_dict('index')
    return d_edgelst, dct_nodes

def create_graph(d_edgelst, dct_nodes,): 

    G = nx.from_pandas_edgelist(
        d_edgelst,
        'n1',
        'n2', 
        edge_attr=['weight', 'weight_abs'])

    # assign size information
    for key, val in dct_nodes.items():
        G.nodes[key]['size'] = val['size']

    # label dict
    labeldict = {}
    for i in G.nodes(): 
        labeldict[i] = i
    
    return G, labeldict

##### PLOT PARAMETERS ######
n_nodes, n_nan, n_rows = 20, 5, 455
A = np.loadtxt('../data/mdl_final/cleaned_nrows_455_maxna_5.dat_params.dat')
n_J = int(n_nodes*(n_nodes-1)/2)
J = A[:n_J] 
h = A[n_J:]

## make it 1-20, and cross-reference that with the related question IDs. 
d_edgelst, dct_nodes = node_edge_lst(n_nodes, J, h)
d_edgelst = d_edgelst.assign(weight_abs = lambda x: np.abs(x['weight']))

## try with thresholding 
d_edgelst_sub = d_edgelst[d_edgelst['weight_abs'] > 0.15]
G, labeldict = create_graph(d_edgelst_sub, dct_nodes)

# setup 
seed = 1
cmap = plt.cm.coolwarm
cutoff_n = 15

# position
pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")

## a few manual tweaks 
x, y = pos[16]
pos[16] = (x-25, y+0)
x, y = pos[7]
pos[7] = (x-10, y+0)
x, y = pos[1]
pos[1] = (x+5, y+5)
x, y = pos[4]
pos[4] = (x+25, y+25)

## plot 
fig, ax = plt.subplots(figsize = (6, 6), facecolor = 'w', dpi = 500)
plt.axis('off')

size_lst = list(nx.get_node_attributes(G, 'size').values())
weight_lst = list(nx.get_edge_attributes(G, 'weight').values())
threshold = sorted([np.abs(x) for x in weight_lst], reverse=True)[cutoff_n]
weight_lst_filtered = [x if np.abs(x)>threshold else 0 for x in weight_lst]

# vmin, vmax edges
vmax_e = np.max(list(np.abs(weight_lst)))
vmin_e = -vmax_e

# vmin, vmax nodes
vmax_n = np.max(list(np.abs(size_lst)))
vmin_n = -vmax_n

size_abs = [abs(x)*3000 for x in size_lst]
weight_abs = [abs(x)*15 for x in weight_lst_filtered]

nx.draw_networkx_nodes(
    G, pos, 
    node_size = 600,#size_abs, 
    node_color = size_lst, 
    edgecolors = 'black',
    linewidths = 0.5,
    cmap = cmap, vmin = vmin_n, vmax = vmax_n 
)
nx.draw_networkx_edges(
    G, pos,
    width = weight_abs, 
    edge_color = weight_lst, 
    alpha = 0.7, # hmmm
    edge_cmap = cmap, edge_vmin = vmin_e, edge_vmax = vmax_e)
nx.draw_networkx_labels(G, pos, font_size = 14, labels = labeldict)
# add to axis
sm_edge = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin_e, vmax=vmax_e))
sm_edge._A = []
sm_node = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin_n, vmax=vmax_n))
sm_node._A = []
axis = plt.gca()
# maybe smaller factors work as well, but 1.1 works fine for this minimal example
#axis.set_xlim([1.1*x for x in axis.get_xlim()])
#axis.set_ylim([1.1*y for y in axis.get_ylim()])
plt.subplots_adjust(bottom=0.1, right=1, left=0, top=1)
#ax_edge = plt.axes([0.95, 0.12, 0.04, 0.74])
ax_edge = plt.axes([0.05, 0, 0.90, 0.05])
ax_node = plt.axes([0.05, -0.2, 0.9, 0.05])
plt.colorbar(sm_edge, cax = ax_edge, orientation='horizontal')
plt.colorbar(sm_node, cax = ax_node, orientation='horizontal')

#cbar.ax.yaxis.set_ticks_position('left') #yaxis.tick_left()
ax.text(0.24, -0.03, r'Pairwise couplings (J$_{ij}$)', size=20, transform=ax.transAxes)
ax.text(0.3, -0.25, r'Local fields (h$_i$)', size = 20, transform = ax.transAxes)
plt.savefig('../fig/parameters.pdf', bbox_inches='tight')

## do it for correlations and means 
## only for FULL records 
d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')
len(d_likelihood['entry_id'].unique()) # 407 total
d_clean = d_likelihood[d_likelihood['p_norm'] > 0.9999] # 171 full obs
## find the configs 
allstates = bin_states(n_nodes) 
clean_configs = d_clean['p_ind'].tolist()
mat_configs = allstates[clean_configs]
## to dataframe 
question_reference = pd.read_csv('../data/analysis/question_reference.csv')
question_reference['question_id'] = question_reference.index + 1 # should be done earlier
question_ids = question_reference['question_id'].to_list() 
df_configs = pd.DataFrame(mat_configs, columns = question_ids)
## correlations 
param_corr = df_configs.corr(method='pearson')
param_corr['node_x'] = param_corr.index
param_corr_melt = pd.melt(param_corr, id_vars = 'node_x', value_vars = question_ids, value_name = 'weight', var_name = 'node_y')
param_corr_melt = param_corr_melt[param_corr_melt['node_x'] < param_corr_melt['node_y']]
## means 
param_mean = df_configs.mean().reset_index(name = 'mean')

## create network 
# create network
G = nx.from_pandas_edgelist(param_corr_melt,
                            'node_x',
                            'node_y',
                            'weight')

# add all node information
for idx, row in param_mean.iterrows(): 
    question_id = row['index']
    G.nodes[question_id]['ID'] = question_id # sanity
    G.nodes[question_id]['size'] = row['mean']

## plot it 


## plot 
fig, ax = plt.subplots(figsize = (6, 6), facecolor = 'w', dpi = 500)
plt.axis('off')

size_lst = list(nx.get_node_attributes(G, 'size').values())
weight_lst = list(nx.get_edge_attributes(G, 'weight').values())
threshold = sorted([np.abs(x) for x in weight_lst], reverse=True)[cutoff_n]
weight_lst_filtered = [x if np.abs(x)>threshold else 0 for x in weight_lst]

# vmin, vmax edges
vmax_e = np.max(list(np.abs(weight_lst)))
vmin_e = -vmax_e

# vmin, vmax nodes
vmax_n = np.max(list(np.abs(size_lst)))
vmin_n = -vmax_n

weight_abs = [abs(x)*20 for x in weight_lst_filtered]

nx.draw_networkx_nodes(
    G, pos, 
    node_size = 700,#size_abs, 
    node_color = size_lst, 
    edgecolors = 'black',
    linewidths = 0.5,
    cmap = cmap, vmin = vmin_n, vmax = vmax_n 
)
nx.draw_networkx_edges(
    G, pos,
    width = weight_abs, 
    edge_color = weight_lst, 
    alpha = 0.7, # hmmm
    edge_cmap = cmap, edge_vmin = vmin_e, edge_vmax = vmax_e)
nx.draw_networkx_labels(G, pos, font_size = 14, labels = labeldict)
# add to axis
sm_edge = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin_e, vmax=vmax_e))
sm_edge._A = []
sm_node = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin_n, vmax=vmax_n))
sm_node._A = []
axis = plt.gca()
# maybe smaller factors work as well, but 1.1 works fine for this minimal example
#axis.set_xlim([1.1*x for x in axis.get_xlim()])
#axis.set_ylim([1.1*y for y in axis.get_ylim()])
plt.subplots_adjust(bottom=0.1, right=1, left=0, top=1)
#ax_edge = plt.axes([0.95, 0.12, 0.04, 0.74])
ax_edge = plt.axes([0.05, 0, 0.90, 0.05])
ax_node = plt.axes([0.05, -0.2, 0.9, 0.05])
plt.colorbar(sm_edge, cax = ax_edge, orientation='horizontal')
plt.colorbar(sm_node, cax = ax_node, orientation='horizontal')

#cbar.ax.yaxis.set_ticks_position('left') #yaxis.tick_left()
ax.text(0.25, -0.03, r"Pearson's correlation", size=20, transform=ax.transAxes)
ax.text(0.43, -0.25, r'Mean', size = 20, transform = ax.transAxes)
plt.savefig('../fig/observation.pdf', bbox_inches='tight')

## match the number of edges. 
## 







########### OLD SHIT #############
'''
def plot_corr(G, labeldict, threshold, n_nodes, tol, seed, outpath): 
    # plot basics
    ## prep
    seed = seed
    cmap = plt.cm.seismic
    ## setup
    fig, axis = plt.subplots(figsize=(7, 5), facecolor = 'w', edgecolor = 'k')
    plt.axis('off')
    ## pos 
    pos = nx.spring_layout(G, seed = seed)
    ## extract values
    size_lst = list(nx.get_node_attributes(G, 'size').values())
    weight_lst = list(nx.get_edge_attributes(G, 'w').values())
    ## with cut-off 
    weight_lst = [x if np.abs(x) > threshold else 0 for x in weight_lst]
    ## color scaling
    vmax = np.max(list(np.abs(size_lst)) + list(np.abs(weight_lst)))
    vmin = -vmax
    ## scale values 
    size_abs = [abs(x)*1500 for x in size_lst]
    weight_abs = [abs(x)*15 for x in weight_lst]
    ## edges
    nx.draw_networkx_nodes(
        G, pos, 
        node_size = size_abs, 
        node_color = size_lst, 
        edgecolors = 'black',
        linewidths = 0.5,
        cmap = cmap, vmin=vmin, vmax=vmax)
    ## nodes
    nx.draw_networkx_edges(
        G, pos,
        width = weight_abs, 
        edge_color = weight_lst, 
        alpha = 0.5, # hmmm
        edge_cmap = cmap, edge_vmin = vmin, edge_vmax = vmax)
    ## labels 
    #label_options = {'edgecolor': 'none', 'facecolor': 'white', 'alpha': 0.5}
    nx.draw_networkx_labels(
        G, pos, 
        labels = labeldict, 
        font_size = 20,
        font_color='whitesmoke')
    ## colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    
    axis = plt.gca()
    # maybe smaller factors work as well, but 1.1 works fine for this minimal example
    #axis.set_xlim([1.1*x for x in axis.get_xlim()])
    #axis.set_ylim([1.1*y for y in axis.get_ylim()])
    
    plt.colorbar(sm, fraction = 0.035)
    out = os.path.join(outpath, f'_nnodes_{n_nodes}_tol_{tol}_threshold_{threshold}.jpeg')
    plt.savefig(f'{out}')

def main(threshold, inpath, outpath): 
    ## setup
    seed = 12 
    #inpath = 'data/DRH/mdl'
    filetype = '*.txt'
    globpath = os.path.join(inpath, filetype)
    filelst = glob.glob(globpath)
    regex_pattern = 'nodes_(\d+)_maxna_(\d+)'
    for file in filelst: 
        p = re.search(regex_pattern, file)
        nh, tol = int(p[1]), int(p[2])
        nJ = int(nh*(nh-1)/2)
        A = np.loadtxt(file, delimiter = ',')
        J_ = A[:nJ]
        h_ = A[nJ:]
        J_edgelst, h_nodes = node_edge_lst(nh, J_, h_)
        G, labeldict = create_graph(J_edgelst, h_nodes)
        plot_corr(
            G = G, 
            labeldict = labeldict,  
            threshold = threshold,
            n_nodes = nh, 
            tol = tol,
            seed = seed,
            outpath = outpath 
        )

####### PLOT DATA STATES #######
import numpy as np 
from fun import top_n_idx, hamming_distance, bin_states
import pandas as pd 
from sklearn.manifold import MDS

# setup
n_rows, n_nan, n_nodes, n_top_states = 455, 5, 20, 150

# load stuff
p = np.loadtxt(f'../data/analysis/p_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
#allstates = np.loadtxt(f'../data/analysis/allstates_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
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
#A = nx.to_agraph(G)

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
len(H.nodes())
'''