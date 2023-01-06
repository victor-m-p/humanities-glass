# COGSCI23
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 
import numpy as np 

def plot_net(d, n, graph_type, suptitle, filename, pos = False): 
    
    # n = 2
    d = d.sort_values('weight').groupby('config_from').tail(n)

    # try to just plot this as is ...
    G = nx.from_pandas_edgelist(d,
                                source = 'config_from',
                                target = 'config_to',
                                edge_attr = 'weight',
                                create_using=graph_type)

    # only the main component
    if graph_type == nx.DiGraph:
        H = G.to_undirected()
        Gcc = sorted(nx.connected_components(H), key=len, reverse=True)
    else: 
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    # edgeweight 
    edge_weight = dict(nx.get_edge_attributes(G, 'weight'))
    edge_list = []
    edge_w = []
    for x, y in edge_weight.items(): 
        edge_list.append(x)
        edge_w.append(y)
        
    # degree 
    degree = dict(G.degree())
    node_list = []
    node_deg = []
    for x, y in degree.items(): 
        node_list.append(x)
        node_deg.append(y)

    # position 
    if not pos: 
        n = len(G.nodes())
        pos = nx.spring_layout(G, weight = 'weight',
                            k = 1/np.sqrt(n),
                            seed = 4)

    # plot 
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
    plt.suptitle(f'{suptitle}', size = large_text)
    plt.savefig(f'../fig/COGSCI23/networks/{filename}.pdf')
    
def get_pos(d, n, graph_type, bias = 1): 
    d = d.sort_values('weight').groupby('config_from').tail(n)

    G = nx.from_pandas_edgelist(d,
                                source = 'config_from',
                                target = 'config_to',
                                edge_attr = 'weight',
                                create_using=graph_type)

    n = len(G.nodes())
    pos = nx.spring_layout(G, weight = 'weight',
                           k = bias/np.sqrt(n),
                           seed = 4)
    return pos 

# setup 
small_text, large_text = 12, 18

# load edgelist (bit funky concept)
d_edgelist = pd.read_csv('../data/COGSCI23/evo_clean/overview.csv')
d_edgelist_10 = d_edgelist[d_edgelist['t_to'] == 10]
d_edgelist_10_max = d_edgelist_10.groupby(['config_from', 'config_to']).size().reset_index(name = 'weight')

# n = 1 
plot_net(d = d_edgelist_10_max,
         n = 1,
         graph_type = nx.Graph, 
         suptitle = 'Undirected (n = 1)',
         filename = 't10_n1_undirected')

plot_net(d = d_edgelist_10_max,
         n = 1,
         graph_type = nx.DiGraph, 
         suptitle = 'Directed (n = 1)',
         filename = 't10_n1_directed')

pos_undirected_1 = get_pos(d = d_edgelist_10_max,
                           n = 1,
                           graph_type = nx.Graph,
                           bias = 1)

plot_net(d = d_edgelist_10_max,
         n = 1, 
         graph_type = nx.DiGraph, 
         suptitle = 'Mixed (n = 1)',
         filename = 't10_n1_mixed',
         pos = pos_undirected_1)

# n = 2
plot_net(d = d_edgelist_10_max,
         n = 2, 
         graph_type = nx.Graph, 
         suptitle = 'Undirected (n = 2)',
         filename = 't10_n2_undirected')

plot_net(d = d_edgelist_10_max,
         n = 2, 
         graph_type = nx.DiGraph, 
         suptitle = 'Directed (n = 2)',
         filename = 't10_n2_directed')

pos_undirected_2 = get_pos(d = d_edgelist_10_max,
                           n = 2,
                           graph_type = nx.Graph,
                           bias = 1)
plot_net(d = d_edgelist_10_max,
         n = 2,
         graph_type = nx.DiGraph,
         suptitle = 'Mixed (n = 2)',
         filename = 't10_n2_mixed',
         pos = pos_undirected_2)


########## labeling ###########

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