import numpy as np 
from fun import top_n_idx, hamming_distance
import pandas as pd 
from sklearn.manifold import MDS
import networkx as nx
import matplotlib.pyplot as plt

# setup
n_rows, n_nan, n_nodes, seed = 455, 5, 20, 254

# load stuff
p = np.loadtxt(f'../data/analysis/p_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
allstates = np.loadtxt(f'../data/analysis/allstates_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')

n_top_states = 10
d_ind = top_n_idx(n_top_states, p, 'p_ind', 'p_val') 
p_ind = d_ind['p_ind'].tolist()
p_val = d_ind['p_val'].tolist()
top_states = allstates[p_ind]

h_distances = hamming_distance(top_states) 

# networkx
idx = [f'hamming{x}' for x in range(10)]
d = pd.DataFrame(h_distances, columns = idx)
d['node_x'] = d.index
d = pd.wide_to_long(d, stubnames = "hamming", i = 'node_x', j = 'node_y').reset_index()
d = d[d['node_x'] != d['node_y']]
d = d.drop_duplicates() 
d = d.assign(weight = lambda x: 1/x['hamming'])

G = nx.from_pandas_edgelist(d, 'node_x', 'node_y', 'weight')
pos = nx.spring_layout(G, weight = 'weight')

labeldict = {}
for i in G.nodes(): 
    labeldict[i] = i

fig, ax = plt.subplots(facecolor = 'w', edgecolor = 'k', dpi = 300)
plt.axis('off')

nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_labels(
    G, pos,
    labels = labeldict, 
    font_size = 20,
    font_color='black')
plt.show();

