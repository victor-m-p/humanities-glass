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
import io 

#inpath = 'data/DRH/mdl'
#filetype = '*.txt'
#globpath = os.path.join(inpath, filetype)
#filelst = glob.glob(globpath)
#x = filelst[0]
#regex_pattern = 'n_(\d+)_tol_(\d.\d+)'
#nh = int(re.search(regex_pattern, x)[1])
#nJ = int(nh*(nh-1)/2)
#A = np.loadtxt(x, delimiter = ',')
#J = A[:nJ] 
#h = A[nJ:]

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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--threshold', required = False, default = 0.0, type = float, help = 'show only meaningful edges')
    ap.add_argument('-i', '--inpath', required = True, type = str, help = 'path to simulated data')
    ap.add_argument('-o', '--outpath', required = True, type = str, help = 'path to save figs')
    args = vars(ap.parse_args())
    main(
        threshold = args['threshold'],
        inpath = args['inpath'],
        outpath = args['outpath'])