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
lst1 = [1, 2, 3]
lst2 = [5, 10]
np.max(lst1+lst2)
def plot_corr(G, labeldict, type, n_nodes, h_scale, J_scale, seed, outpath): 
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
    ## color scaling
    if type == 'MPF': 
        vmax = np.max(size_lst + weight_lst)
        vmin = np.min(size_lst + weight_lst)
    else: 
        vmin = -1
        vmax = 1
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
    out = os.path.join(outpath, f'type_{type}_nnodes_{n_nodes}_hscale_{h_scale}_Jscape_{J_scale}.jpeg')
    plt.savefig(f'{out}')

def main(inpath, outpath): 
    ## setup
    seed = 12 
    filetype = '*.txt'
    globpath = os.path.join(inpath, filetype)
    filelst = glob.glob(f'{globpath}')

    ## all combinations of simulation settings
    sim_combinations = []
    regex_pattern = '(\d+)_hscale_(\d.\d+)_Jscale_(\d.\d+)_nsamp_(\d+)_seed_(\d+)'
    for file in filelst: 
        p = re.search(regex_pattern, file)
        nn, hs, js, ns, s = int(p[1]), float(p[2]), float(p[3]), int(p[4]), int(p[5])
        sim_combinations.append([nn, hs, js, ns, s])
    ## unique combinations of params
    sim_combinations = Counter([tuple(i) for i in sim_combinations])
    ## loop over pipeline 
    for sim_combination in sim_combinations: 
        nn, hs, js, ns, s = sim_combination
        for type in ['MPF', 'naive']:
            if type == 'MPF': 
                h_means, J_corr = ['h', 'J']
            else: 
                h_means, J_corr = ['means', 'corr']
            h_string = f'{h_means}_nnodes_{nn}_hscale_{hs}_Jscale_{js}_nsamp_{ns}_seed_{s}.txt'
            J_string = f'{J_corr}_nnodes_{nn}_hscale_{hs}_Jscale_{js}_nsamp_{ns}_seed_{s}.txt'
            h_path = os.path.join(inpath, h_string)
            J_path = os.path.join(inpath, J_string)
            h_ = np.loadtxt(h_path)
            J_ = np.loadtxt(J_path)
            J_edgelst, h_nodes = node_edge_lst(nn, J_, h_)
            G, labeldict = create_graph(J_edgelst, h_nodes)
            plot_corr(
                G = G, 
                labeldict = labeldict, 
                type = type, 
                n_nodes = nn, 
                h_scale = hs,
                J_scale = js,
                seed = seed,
                outpath = outpath 
            )        

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inpath', required = True, type = str, help = 'path to simulated data')
    ap.add_argument('-o', '--outpath', required = True, type = str, help = 'path to save figs')
    args = vars(ap.parse_args())
    main(
        inpath = args['inpath'],
        outpath = args['outpath'])