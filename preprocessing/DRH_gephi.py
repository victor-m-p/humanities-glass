import re 
import numpy as np
import glob 
import os 
import itertools 
import pandas as pd 
import networkx as nx 

inpath = '../data/mdl/params_nodes_20_maxna_7.txt'
outpath = '../data/network'

regex_pattern = 'nodes_(\d+)_maxna_(\d+)'
nnan = int(re.search(regex_pattern, inpath)[2])
nh = int(re.search(regex_pattern, inpath)[1])
nJ = int(nh*(nh-1)/2)
A = np.loadtxt(inpath, delimiter = ',')
J = A[:nJ] 
h = A[nJ:]

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
        G.nodes[key]['label'] = key
    
    return G

J_edgelst, h_nodes = node_edge_lst(nh, J, h)
G = create_graph(J_edgelst, h_nodes)

# write the file
outname = f'gephi_nodes_{nh}_maxna_{nnan}.gexf'
out = os.path.join(outpath, outname)
nx.write_gexf(G, out)