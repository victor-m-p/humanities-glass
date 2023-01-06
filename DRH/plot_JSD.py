# COGSCI23
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 
import numpy as np 

# load JSD 
d_JSD = pd.read_csv('../data/COGSCI23/evo_entropy/JSD_10.csv')
d_JSD['weight'] = d_JSD['JSD']
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

# plot 2: force spread
pos = nx.spring_layout(G, weight = 'weight',
                       k = 0.2/np.sqrt(260),
                       seed = 4)
nx.draw_networkx_nodes(G, pos, 
                       nodelist = node_list, 
                       node_size = [x*0.01 for x in node_size],
                       node_color = node_color)
nx.draw_networkx_edges(G, pos, 
                       edgelist = edge_list,
                       width = [x*0.01 for x in edge_w])

# plot 3: more spread
pos = nx.spring_layout(G, weight = 'weight',
                       k = 0.15/np.sqrt(260),
                       seed = 12)
nx.draw_networkx_nodes(G, pos, 
                       nodelist = node_list, 
                       node_size = [x*0.01 for x in node_size],
                       node_color = node_color)
nx.draw_networkx_edges(G, pos, 
                       edgelist = edge_list,
                       width = [x*0.001 for x in edge_w])

# find some of the nodes that we know 
config_ids = nx.get_node_attributes(G, 'config_id')
config_ids = {key:int(val) for key, val in config_ids.items()}
labels = {}
for node, attr in G.nodes(data=True): 
    nsize = attr['nodesize']
    if nsize > 1000: 
        labels[node] = int(attr['config_id'])
    else: 
        labels[node] = ""
labels    
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
nx.draw_networkx_labels(G, pos, labels)

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

# actually run for 100 time-steps and run all 261. 