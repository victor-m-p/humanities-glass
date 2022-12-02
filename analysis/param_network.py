import itertools 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import networkx as nx 
import glob
from collections import Counter  
import re
import numpy as np
import argparse 

n_nodes, n_nan = 20, 5
A = np.loadtxt('../data/mdl_final/cleaned_nrows_455_maxna_5.dat_params.dat')
n_J = int(n_nodes*(n_nodes-1)/2)
J = A[:n_J] 
h = A[n_J:]

## try for correlation data
def node_edge_lst(n, corr_J, means_h): 
    nodes = [node for node in range(n)]
    comb = list(itertools.combinations(nodes, 2))
    d_edgelst = pd.DataFrame(comb, columns = {'n1', 'n2'})
    d_edgelst['w'] = corr_J
    d_nodes = pd.DataFrame(nodes, columns = {'n'})
    d_nodes['size'] = means_h
    d_nodes = d_nodes.set_index('n')
    dct_nodes = d_nodes.to_dict('index')
    return d_edgelst, dct_nodes

d_edgelst, dct_nodes = node_edge_lst(n_nodes, J, h)
d_edgelst = d_edgelst.assign(w_abs = lambda x: np.abs(x['w']))

## create graph 
def create_graph(d_edgelst, dct_nodes): 

    G = nx.from_pandas_edgelist(
        d_edgelst,
        'n1',
        'n2', 
        edge_attr=['w', 'w_abs'])

    # assign size information
    for key, val in dct_nodes.items():
        G.nodes[key]['size'] = val['size']

    # label dict
    labeldict = {}
    for i in G.nodes(): 
        labeldict[i] = i
    
    return G, labeldict

G, labeldct = create_graph(d_edgelst, dct_nodes)
seed = 32
threshold = 0.4
cmap = plt.cm.coolwarm
fig, ax = plt.subplots(figsize = (7, 5), facecolor = 'w')
plt.axis('off')
pos = nx.spring_layout(G, weight = 'w_abs', seed = seed)
size_lst = list(nx.get_node_attributes(G, 'size').values())
weight_lst = list(nx.get_edge_attributes(G, 'w').values())
weight_lst_filtered = [x if np.abs(x)>threshold else 0 for x in weight_lst]
vmax = np.max(list(np.abs(size_lst)) + list(np.abs(weight_lst)))
vmin = -vmax
size_abs = [abs(x)*1500 for x in size_lst]
weight_abs = [abs(x)*15 for x in weight_lst_filtered]
nx.draw_networkx_nodes(
    G, pos, 
    node_size = size_abs, 
    node_color = size_lst, 
    edgecolors = 'black',
    linewidths = 0.5,
    cmap = cmap, vmin = vmin, vmax = vmax 
)
nx.draw_networkx_edges(
    G, pos,
    width = weight_abs, 
    edge_color = weight_lst, 
    alpha = 0.5, # hmmm
    edge_cmap = cmap, edge_vmin = vmin, edge_vmax = vmax)
## labels 
#label_options = {'edgecolor': 'none', 'facecolor': 'white', 'alpha': 0.5}
#nx.draw_networkx_labels(
#    G, pos, 
#    labels = labeldct, 
#    font_size = 20,
#    font_color='whitesmoke')
## colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
sm._A = []

axis = plt.gca()
# maybe smaller factors work as well, but 1.1 works fine for this minimal example
#axis.set_xlim([1.1*x for x in axis.get_xlim()])
#axis.set_ylim([1.1*y for y in axis.get_ylim()])

plt.colorbar(sm, fraction = 0.035)
plt.show();


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

# simons plot 
import numpy as np 
from fun import top_n_idx, hamming_distance
import pandas as pd 
from sklearn.manifold import MDS

# setup
n_rows, n_nan, n_nodes, n_top_states = 455, 5, 20, 150

# load stuff
p = np.loadtxt(f'../data/analysis/p_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
allstates = np.loadtxt(f'../data/analysis/allstates_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')

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

G = nx.from_pandas_edgelist(d,
                            'node_x',
                            'node_y',
                            'hamming')

node_attr = d_ind.to_dict('index')
for idx, val in node_attr.items(): 
    for attr in val: 
        idx = val['index']
        G.nodes[idx][attr] = val[attr]

# take out weight
node_size = nx.get_node_attributes(G, 'p_val')

# take out edge weight
edge_weight = nx.get_edge_attributes(G, 'hamming')

# simon scaled this by something funny..
# also, consider sorting 

# to lists
node_size_lst = list(node_size.values())
edge_weight_lst = list(edge_weight.values()) 

# scale lists
node_scaling = 5000
node_size_lst_scaled = [x*node_scaling for x in node_size_lst]

# position
## unweighted, so should not matter
n_nodes = len(G.nodes())
scalar = 0.2
iter = 50
k = scalar/np.sqrt(n_nodes)
pos = nx.spring_layout(G, k = k, iterations = iter, weight = 'hamming', seed = seed)
pos = nx.kamada_kawai_layout(G, weight = 'hamming')
pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
cmap = plt.cm.Blues
# plot it 
fig, ax = plt.subplots(facecolor = 'w', dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, 
                       node_size = node_size_lst_scaled, 
                       node_color = node_size_lst_scaled, 
                       linewidths = 0.5, edgecolors = 'black',
                       cmap = cmap)
nx.draw_networkx_edges(G, pos, width = 1, alpha = 0.5,
                       edge_cmap = cmap)

# https://github.com/bhargavchippada/forceatlas2

## plotting with graphviz
#A = nx.to_agraph(G)
