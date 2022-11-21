import pandas as pd 
import numpy as np 
from sim_fun import p_dist, bin_states
import itertools 
import matplotlib.pyplot as plt
import os
import networkx as nx 

# setup
s = 10000 # can we do exact instead of sample based for corr and means?
n = 5
scale = 1.0
outpath = 'figs'

# data generation
np.random.seed(124)
h = np.random.normal(scale=scale, size=n)
J = np.random.normal(scale=scale, size=n*(n-1)//2)
hJ = np.concatenate((h, J))

# sample based on J, h
p = p_dist(h, J) # probability of all states
allstates = bin_states(n)
sample = allstates[np.random.choice(range(2**n), # doesn't have to be a range
                                    size=s, # how many samples
                                    replace=True, # a value can be selected multiple times
                                    p=p)] 

# get raw correlations and raw mean 
## does not make a difference for corr whether
## it is coded as -1 or 0 I think. 
corr = np.corrcoef(sample.T)
m = corr.shape[0]
corr = corr[np.triu_indices(m, 1)]
means = np.mean(sample, axis = 0)

## how do we generate data based on this?
## i.e. can we show that we cannot reproduce
## reasonable data from this?
corr
means

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

corr_d_edgelst, corr_dct_nodes = node_edge_lst(n, corr, means)
MPF_d_edgelst, MPF_dct_nodes = node_edge_lst(n, J, h)

## create graph 
def create_graph(d_edgelst, dct_nodes): 

    G = nx.from_pandas_edgelist(
        d_edgelst,
        'n1',
        'n2', 
        edge_attr='w')

    # assign size information
    for key, val in dct_nodes.items():
        G.nodes[key]['size'] = val['size']

    # label dict
    labeldict = {}
    for i in G.nodes(): 
        labeldict[i] = i
    
    return G, labeldict

corr_G, corr_labeldict = create_graph(corr_d_edgelst, corr_dct_nodes)
MPF_G, MPF_labeldict = create_graph(MPF_d_edgelst, MPF_dct_nodes)

def plot_corr(G, labeldict, type, n, scale, outpath): 
    # plot basics
    ## prep
    seed = 10
    cmap = plt.cm.seismic
    vmin = -1
    vmax = 1
    ## setup
    fig, axis = plt.subplots(figsize=(7, 5), facecolor = 'w', edgecolor = 'k')
    plt.axis('off')
    ## pos 
    pos = nx.spring_layout(G, seed = seed)
    ## extract values
    size_lst = list(nx.get_node_attributes(G, 'size').values())
    weight_lst = list(nx.get_edge_attributes(G, 'w').values())
    ## scale values 
    size_abs = [abs(x)*1500 for x in size_lst]
    weight_abs = [abs(x)*15 for x in weight_lst]
    ## edges
    nx.draw_networkx_nodes(
        G, pos, 
        node_size = size_abs, 
        node_color = size_lst, 
        cmap = cmap, vmin=vmin, vmax=vmax)
    ## nodes
    nx.draw_networkx_edges(
        G, pos, 
        width = weight_abs, 
        edge_color = weight_lst, 
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
    axis.set_xlim([1.1*x for x in axis.get_xlim()])
    axis.set_ylim([1.1*y for y in axis.get_ylim()])
    
    plt.colorbar(sm, fraction = 0.035)
    plt.suptitle(f'{type}', fontsize = 20) # should be in the middle
    out = os.path.join(outpath, f'type_{type}_n_{n}_scale_{scale}.jpeg')
    plt.savefig(f'{out}')

plot_corr(
    G = corr_G, 
    labeldict = corr_labeldict, 
    type = 'CORR', 
    n = n,
    scale = scale,
    outpath = outpath)

plot_corr(
    G = MPF_G,
    labeldict = MPF_labeldict,
    type = 'MPF',
    n = n,
    scale = scale,
    outpath = outpath)
