# COGSCI23
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 
import numpy as np 

small_text, large_text = 12, 18

# load JSD 
d_JSD = pd.read_csv('../data/COGSCI23/evo_entropy/JSD_10.csv')
d_JSD['weight'] = 1-d_JSD['JSD']
d_JSD = d_JSD[['i', 'j', 'weight']]

# load GMM
d_GMM = pd.read_csv('../data/COGSCI23/evo_GMM/t_10_att_from.csv')
d_GMM['node'] = d_GMM.index

# load edgelist (bit funky concept)
d_edgelist = pd.read_csv('../data/COGSCI23/evo_clean/overview.csv')
d_from = d_edgelist[['config_from']].drop_duplicates()
d_from = d_from.rename(columns = {'config_from': 'config_to'})
d_test = d_edgelist.merge(d_from, on = 'config_to', how = 'inner')
d_test = d_test.groupby('config_to').size().reset_index(name = 'count')

# overall nodeObj
d_test = d_test.rename(columns = {'config_to': 'config_from'})
d_nodeattr = d_GMM.merge(d_test, on = 'config_from', how = 'left').fillna(0)

# create NETWORKX obj 
G = nx.from_pandas_edgelist(d_JSD, 
                            source = 'i',
                            target = 'j',
                            edge_attr = 'weight')

# add node attributes
for num, row in d_nodeattr.iterrows():
    nodesize = row['count']
    node_id = row['node']
    cluster = row['cluster']
    config_id = row['config_from']
    G.nodes[node_id]['cluster'] = cluster 
    G.nodes[node_id]['config_id'] = config_id
    G.nodes[node_id]['nodesize'] = nodesize

# prepare plot 
## edgeweight 
edge_weight = dict(nx.get_edge_attributes(G, 'weight'))
edge_list = []
edge_w = []
for x, y in edge_weight.items(): 
    edge_list.append(x)
    edge_w.append(y)
    
## degree 
degree = dict(G.degree(weight = 'weight'))
node_list = []
node_deg = []
for x, y in degree.items(): 
    node_list.append(x)
    node_deg.append(y)

## size 
node_count = dict(nx.get_node_attributes(G, 'nodesize')) 
node_size = [x for x in node_count.values()]

## color 
node_cluster = dict(nx.get_node_attributes(G, 'cluster'))
node_color = ['tab:blue' if x == 0 else 'tab:orange' for x in node_cluster.values()]

# plot 1: hairball 
fig, ax = plt.subplots(dpi = 300)
pos = nx.spring_layout(G, weight = 'weight',
                       k = 1/np.sqrt(260),
                       seed = 4)
nx.draw_networkx_nodes(G, pos, 
                       nodelist = node_list, 
                       node_size = [x*0.01 for x in node_size],
                       node_color = node_color)
nx.draw_networkx_edges(G, pos, 
                       edgelist = edge_list,
                       width = [x*0.01 for x in edge_w])

# plot 2: more spread
fig, ax = plt.subplots(dpi = 300)
pos = nx.spring_layout(G, weight = 'weight',
                       k = 0.15/np.sqrt(260),
                       seed = 12)
nx.draw_networkx_nodes(G, pos, 
                       nodelist = node_list, 
                       node_size = [x*0.01 for x in node_size],
                       node_color = node_color)
nx.draw_networkx_edges(G, pos, 
                       edgelist = edge_list,
                       width = [x*0.01 for x in edge_w])


# find interesting states to label
pd.set_option('display.max_colwidth', None)
d_maxlik = pd.read_csv('/home/vpoulsen/humanities-glass/data/analysis/entry_maxlikelihood.csv')
d_maxlik = d_maxlik[['config_id', 'entry_drh']].drop_duplicates()
d_maxlik = d_maxlik.rename(columns = {'config_id': 'config_from'})
d_maxlik = d_maxlik.groupby('config_from')['entry_drh'].unique().reset_index(name = 'religions')
d_labeling = d_maxlik.merge(d_nodeattr, on = 'config_from', how = 'inner')
d_labeling = d_labeling.sort_values('count', ascending = False)

# merge it with some of our old data
# i.e. so that we can reference with our 
# main plot (perhaps also community for those where it applies)
network_landscape = pd.read_csv('../data/analysis/network_information_enriched.csv')
network_landscape = network_landscape[['config_id', 'config_prob', 'node_id', 'comm_color', 'comm_label']]
network_landscape = network_landscape.rename(columns = {'config_id': 'config_from'})
d_labeling = d_labeling.merge(network_landscape, on = 'config_from', how = 'left')
d_labeling.sort_values(['cluster', 'count'], 
                       ascending = [False, False]).head(20)

# network with flow to maximum (directed)

# network with color / size by diameter (or mean dist. from home).

# plot something different 
d_edgelist_10 = d_edgelist[d_edgelist['t_to'] == 10]
d_edgelist_10_max = d_edgelist_10.groupby(['config_from', 'config_to']).size().reset_index(name = 'weight')

# n = 2
d_edgelist_samp = d_edgelist_10_max.sort_values('weight').groupby('config_from').tail(2)

# try to just plot this as is ...
G = nx.from_pandas_edgelist(d_edgelist_samp,
                            source = 'config_from',
                            target = 'config_to',
                            edge_attr = 'weight')

# only the main component
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G = G.subgraph(Gcc[0])

# prepare plot 
## edgeweight 
edge_weight = dict(nx.get_edge_attributes(G, 'weight'))
edge_list = []
edge_w = []
for x, y in edge_weight.items(): 
    edge_list.append(x)
    edge_w.append(y)
    
## degree 
degree = dict(G.degree())
node_list = []
node_deg = []
for x, y in degree.items(): 
    node_list.append(x)
    node_deg.append(y)

n = len(G.nodes())

## plot (takes a while) 
pos = nx.spring_layout(G, weight = 'weight',
                       k = 1/np.sqrt(n),
                       seed = 4)

fig, ax = plt.subplots(dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, 
                       nodelist = node_list, 
                       node_size = [x for x in node_deg],
                       node_color = 'tab:orange',
                       linewidths = 0.5,
                       edgecolors = 'black')
nx.draw_networkx_edges(G, pos, 
                       edgelist = edge_list,
                       width = [x*0.1 for x in edge_w],
                       alpha = [x/max(edge_w)*0.5 for x in edge_w],
                       edge_color = 'tab:blue')
plt.suptitle('Undirected (n=2)', size = large_text)
plt.savefig('../fig/COGSCI23/networks/t10_n2_undirected.pdf')


# n = 1
d_edgelist_samp = d_edgelist_10_max.sort_values('weight').groupby('config_from').tail(1)

# try to just plot this as is ...
G = nx.from_pandas_edgelist(d_edgelist_samp,
                            source = 'config_from',
                            target = 'config_to',
                            edge_attr = 'weight')

# only GCC
#H = G.to_undirected()
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G = G.subgraph(Gcc[0])

# prepare plot 
## edgeweight 
edge_weight = dict(nx.get_edge_attributes(G, 'weight'))
edge_list = []
edge_w = []
for x, y in edge_weight.items(): 
    edge_list.append(x)
    edge_w.append(y)
    
## degree 
degree = dict(G.degree())
node_list = []
node_deg = []
for x, y in degree.items(): 
    node_list.append(x)
    node_deg.append(y)

n = len(G.nodes())

## plot (takes a while) 
pos = nx.spring_layout(G, weight = 'weight',
                       k = 1/np.sqrt(n),
                       seed = 4)

fig, ax = plt.subplots(dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, 
                       nodelist = node_list, 
                       node_size = [x*5 for x in node_deg],
                       node_color = 'tab:orange',
                       linewidths = 0.5,
                       edgecolors = 'black')
nx.draw_networkx_edges(G, pos, 
                       edgelist = edge_list,
                       width = [x*0.05 for x in edge_w],
                       alpha = [x/max(edge_w) for x in edge_w],
                       edge_color = 'tab:blue')
plt.suptitle('Undirected (n=1)', size = large_text)
plt.savefig('../fig/COGSCI23/networks/t10_n1_undirected.pdf')

# directed graph version 
d_edgelist_samp = d_edgelist_10_max.sort_values('weight').groupby('config_from').tail(1)

# try to just plot this as is ...
G = nx.from_pandas_edgelist(d_edgelist_samp,
                            source = 'config_from',
                            target = 'config_to',
                            edge_attr = 'weight',
                            create_using = nx.DiGraph)

# only GCC
H = G.to_undirected()
Gcc = sorted(nx.connected_components(H), key=len, reverse=True)
G = G.subgraph(Gcc[0])

# prepare plot 
## edgeweight 
edge_weight = dict(nx.get_edge_attributes(G, 'weight'))
edge_list = []
edge_w = []
for x, y in edge_weight.items(): 
    edge_list.append(x)
    edge_w.append(y)
    
## degree 
degree = dict(G.degree())
node_list = []
node_deg = []
for x, y in degree.items(): 
    node_list.append(x)
    node_deg.append(y)

n = len(G.nodes())

## plot (takes a while) 
pos = nx.spring_layout(G, weight = 'weight',
                       k = 1/np.sqrt(n),
                       seed = 4)

# plot
fig, ax = plt.subplots(dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, 
                       nodelist = node_list, 
                       node_size = [x*5 for x in node_deg],
                       node_color = 'tab:orange',
                       linewidths = 0.5,
                       edgecolors = 'black')
nx.draw_networkx_edges(G, pos, 
                       edgelist = edge_list,
                       width = [x*0.05 for x in edge_w],
                       alpha = [x/max(edge_w)*0.5 for x in edge_w],
                       edge_color = 'tab:blue')
plt.suptitle('Directed (n=1)', size = large_text)
plt.savefig('../fig/COGSCI23/networks/t10_n1_directed.pdf')

# n = 2 again
d_edgelist_samp = d_edgelist_10_max.sort_values('weight').groupby('config_from').tail(2)

# try to just plot this as is ...
G = nx.from_pandas_edgelist(d_edgelist_samp,
                            source = 'config_from',
                            target = 'config_to',
                            edge_attr = 'weight',
                            create_using = nx.DiGraph)

# only GCC
H = G.to_undirected()
Gcc = sorted(nx.connected_components(H), key=len, reverse=True)
G = G.subgraph(Gcc[0])

# prepare plot 
## edgeweight 
edge_weight = dict(nx.get_edge_attributes(G, 'weight'))
edge_list = []
edge_w = []
for x, y in edge_weight.items(): 
    edge_list.append(x)
    edge_w.append(y)
    
## degree 
degree = dict(G.degree())
node_list = []
node_deg = []
for x, y in degree.items(): 
    node_list.append(x)
    node_deg.append(y)

n = len(G.nodes())

## plot (takes a while) 
pos = nx.spring_layout(G, weight = 'weight',
                       k = 1/np.sqrt(n),
                       seed = 4)

# plot
fig, ax = plt.subplots(dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, 
                       nodelist = node_list, 
                       node_size = [x*5 for x in node_deg],
                       node_color = 'tab:orange',
                       linewidths = 0.5,
                       edgecolors = 'black')
nx.draw_networkx_edges(G, pos, 
                       edgelist = edge_list,
                       width = [x*0.05 for x in edge_w],
                       alpha = [x/max(edge_w) for x in edge_w],
                       edge_color = 'tab:blue')
plt.suptitle('Directed (n=2)', size = large_text)
plt.savefig('../fig/COGSCI23/networks/t10_n2_directed.pdf')

# layout based on undirected but show the directed ones? 
