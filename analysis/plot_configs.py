import matplotlib.pyplot as plt 
from matplotlib.colors import rgb2hex
import networkx as nx 
import numpy as np
import pandas as pd 
from fun import bin_states, top_n_idx, hamming_distance

def draw_network(Graph, pos, cmap_name, alpha, nodelst, nodesize, edgelst, edgesize, ax_idx): 
    cmap = plt.cm.get_cmap(cmap_name)
    nx.draw_networkx_nodes(Graph, pos, 
                           nodelist = nodelst,
                           node_size = nodesize, 
                           node_color = nodesize,
                           linewidths = 0.5, edgecolors = 'black',
                           cmap = cmap,
                           ax = ax[ax_idx])
    cmap = plt.cm.get_cmap(cmap_name, 2)
    rgba = rgb2hex(cmap(1))
    nx.draw_networkx_edges(Graph, pos, width = edgesize, 
                        alpha = alpha, edgelist = edgelst,
                        edge_color = rgba,
                        #edge_color = edgesize,
                        #edge_cmap = cmap,
                        ax = ax[ax_idx])
    ax[ax_idx].set_axis_off()

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

def hamming_edges(n_top_states, h_distances):
    idx = [f'hamming{x}' for x in range(n_top_states)]
    d = pd.DataFrame(h_distances, columns = idx)
    d['node_x'] = d.index
    d = pd.wide_to_long(d, stubnames = "hamming", i = 'node_x', j = 'node_y').reset_index()
    d = d[d['node_x'] != d['node_y']] # remove diagonal 
    d = d.drop_duplicates() # remove duplicates
    return d 

# assign weight information to G 
def edge_strength(G, nodestrength): 
    Gcopy = G.copy()
    for edge_x, edge_y in Gcopy.edges():
        pmass_x = Gcopy.nodes[edge_x][nodestrength]
        pmass_y = Gcopy.nodes[edge_y][nodestrength]
        pmass_mult = pmass_x*pmass_y 
        pmass_add = pmass_x+pmass_y
        Gcopy.edges[(edge_x, edge_y)]['pmass_mult'] = pmass_mult 
        Gcopy.edges[(edge_x, edge_y)]['pmass_add'] = pmass_add  
    return Gcopy 

# setup
n_rows, n_nan, n_nodes, n_top_states = 455, 5, 20, 150

# load data
p = np.loadtxt(f'../data/analysis/p_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')
nodes_reference = pd.read_csv(f'../data/analysis/nref_nrows_455_maxna_5_nodes_20.csv')

# bin states and get likelihood and index
allstates = bin_states(n_nodes) 
d_ind = top_n_idx(n_top_states, p, 'p_ind', 'p_raw') 
d_ind['node_id'] = d_ind.index # 150

## add likelihood information for the states that appear in top states
def datastate_information(d_likelihood, nodes_reference, d_ind): 
    # merge with nodes reference to get entry_name
    d_likelihood = d_likelihood[['entry_id', 'p_ind', 'p_norm']]
    d_likelihood = d_likelihood.merge(nodes_reference, on = 'entry_id', how = 'inner')
    # make sure that dtypes are preserved 
    d_ind = d_ind.convert_dtypes()
    d_likelihood = d_likelihood.convert_dtypes()
    # merge with d_ind to get data-state probability 
    d_likelihood = d_likelihood.merge(d_ind, on = 'p_ind', indicator = True)
    d_likelihood.rename(columns = {'_merge': 'state'}, inplace = True)
    d_likelihood = d_likelihood.replace({'state': {'left_only': 'only_data', 
                                        'right_only': 'only_config',
                                        'both': 'overlap'}})
    # only interested in states both in data and in top configurations
    d_overlap = d_likelihood[d_likelihood['state'] == 'overlap'].drop(columns={'state'})
    # add information about maximum likelihood 
    max_likelihood = d_overlap.groupby('entry_id')['p_norm'].max().reset_index(name = 'p_norm')
    d_overlap = d_overlap.merge(max_likelihood, on = ['entry_id', 'p_norm'], how = 'left', indicator=True)
    d_overlap = d_overlap.rename(columns = {'_merge': 'max_likelihood'})
    d_overlap = d_overlap.replace({'max_likelihood': {'both': 'yes', 'left_only': 'no'}})
    d_overlap['full_record'] = np.where(d_overlap['p_norm'] == 1, 'yes', 'no')
    return d_overlap 

d_overlap = datastate_information(d_likelihood, nodes_reference, d_ind) # 407

## weight for configurations (proportional to data state weight) 
def datastate_weight(d_overlap): 
    d_entry_node = d_overlap[['entry_id', 'node_id']]
    d_datastate_weight = d_entry_node.groupby('entry_id').size().reset_index(name = 'entry_count')
    d_datastate_weight = d_datastate_weight.assign(entry_weight = lambda x: 1/x['entry_count'])
    d_datastate_weight = d_entry_node.merge(d_datastate_weight, on = 'entry_id', how = 'inner')
    d_datastate_weight = d_datastate_weight.groupby('node_id')['entry_weight'].sum().reset_index(name = 'datastate_sum')
    return d_datastate_weight 

d_datastate_weight = datastate_weight(d_overlap) # 129

## labels by node_id
### take the maximum p_norm per node_id
### if there are ties do not break them for now
d_max_weight = d_overlap.groupby('node_id')['p_norm'].max().reset_index(name = 'p_norm')
d_max_weight = d_overlap.merge(d_max_weight, on = ['node_id', 'p_norm'], how = 'inner')
d_max_weight = d_datastate_weight.merge(d_max_weight, on = 'node_id', how = 'inner')

## labels by node_id 
### break ties randomly for now 
def merge_node_attributes(d_max_weight, d_ind): 
    d_datastate_attr = d_max_weight.groupby('node_id').sample(n=1, random_state=421)
    d_datastate_attr = d_datastate_attr.drop(columns = {'p_raw'})
    node_attr = d_ind.merge(d_datastate_attr, on = ['node_id', 'p_ind'], how = 'left', indicator = True)
    node_attr = node_attr.rename(columns = {'_merge': 'datastate'})
    node_attr = node_attr.replace({'datastate': {'both': 'yes', 'left_only': 'no'}})
    # configs that are not datastates, fill na (easier later)
    node_attr['datastate_sum'] = node_attr['datastate_sum'].fillna(0)
    node_attr['max_likelihood'] = node_attr['max_likelihood'].fillna('no')
    #node_attr_dict = node_attr.to_dict('index')
    return node_attr

node_attr = merge_node_attributes(d_max_weight, d_ind)
node_attr_dict = node_attr.to_dict('index')

# hamming distance
p_ind = d_ind['p_ind'].tolist()
top_states = allstates[p_ind]
h_distances = hamming_distance(top_states) 
h_distances = hamming_edges(n_top_states, h_distances)
h_distances = h_distances[h_distances['hamming'] == 1]
 
# create network
G = nx.from_pandas_edgelist(h_distances,
                            'node_x',
                            'node_y',
                            'hamming')

pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")

# add all node information
node_attr_dict
for idx, val in node_attr_dict.items(): 
    for attr in val: 
        idx = val['node_id']
        G.nodes[idx][attr] = val[attr]

# process 
G_full = edge_strength(G, 'p_raw') # fix
edgelst_full, edgew_full = edge_information(G_full, 'pmass_mult', 'hamming', 30000)
nodelst_full, nodesize_full = node_information(G_full, 'p_raw', 5000)

G_data = edge_strength(G, 'datastate_sum')
edgelst_data, edgew_data = edge_information(G_data, 'pmass_mult', 'hamming', 0.2)
nodelst_data, nodesize_data = node_information(G_data, 'datastate_sum', 15)

# plot 
fig, ax = plt.subplots(1, 2, facecolor = 'w', figsize = (14, 8), dpi = 500)
draw_network(G_full, pos, 'Blues', 0.6, nodelst_full, nodesize_full, edgelst_full, edgew_full, 0)
draw_network(G_data, pos, 'Blues', 0.6, nodelst_data, nodesize_data, edgelst_data, edgew_data, 1)
plt.savefig('../fig/configurations.pdf')

# status 
## (1) need to scale "together" somehow (same min and max, or match the mean?) -- both nodes and edges
## (2) need good way of referencing which records we are talking about
## (3) largest differences between the two plots..?
## (4) could try to run community detection as well 

# reference plot 
labeldict = {}
for node in nodelst_data:
    node_id = G.nodes[node]['node_id']
    labeldict[node] = node_id

fig, ax = plt.subplots(1, 2, figsize = (14, 8), dpi = 500)
draw_network(G_full, pos, 'Blues', 0.6, nodelst_full, nodesize_full, edgelst_full, edgew_full, 0)
draw_network(G_data, pos, 'Blues', 0.6, nodelst_data, nodesize_data, edgelst_data, edgew_data, 1)
nx.draw_networkx_labels(G_full, pos, font_size = 6, labels = labeldict, ax = ax[0])
nx.draw_networkx_labels(G_data, pos, font_size = 6, labels = labeldict, ax = ax[1])
plt.savefig('../fig/reference_temp.pdf')

# what are in these clusters?
def get_match(n):
    dm = d_max_weight[d_max_weight['node_id'] == n][['entry_name']]
    print(dm.head(10))


# central configuration (Catholic, Islamic, -- also Egypt -- Axial?)
get_match(0) # 12th-13th C. Cistercians (Catholic), Savigny (12th C. Cistercian), Nahdlatul (Islamic Indonesia), Moravian (Christian), Knights Templar (Roman Cat.), Naqshbandi (sunni sufism), Greek (orthodox), Zapotec (south american, polytheistic)
get_match(1) # Selinous (local religion, e.g. sacrifices, magic), Jesuits Britain (catholic), Confucianism (Chinese)
get_match(2) # Cham Bani (sunni/shia mix), Ancient Egypt, Ugarit, Isis
get_match(4) # Islam in Aceh
get_match(14) # Donatism (christian), Irish Rom. Cat., Opus Dei (Catholic), Sino-Muslims, Mourides (sufist)

# around number three (not sure)
get_match(3) # Branch Davidians (Texas, apocalyptic, seventh-day), Egypt Salafism, Anabaptist Mennonites, Church of Christ, Jehova
get_match(15) # Muslim Ahmadiyya (Islamic/messianic revival, 19th century British India)
get_match(72) # Estado de India Renegades in Deccan (weird mixture)
get_match(13) # Edinoverie (Russian, Christians, outsiders), Calvinism

# right-top valley (Baptist/Methodist/Evangelical)
get_match(18) # Wesleyanism (methodist), Methodists, Friends, Neo-charismatic, No-debt US Evangelicanism
get_match(85) # Manus (pre-Christian Papua New Guinea, spirit belief)
get_match(50) # Sachchai (Nepal Christians, evangelist)
get_match(42) # Southern Baptists (Protestant, Baptist, Evangelical)

# right-bot valley (Protestant + other)
get_match(90) # Nigerian Pentecostalism (progressive, protestant), Anglican Korea (progressive, protestant)
get_match(49) # Early Orthodox (calvinist, protestant), Tijaniyya (Sufi, west/north Africa)
get_match(13) # Edinoverie (Russian, Christians, outsiders), Calvinism

# lower cluster (Hindu/Buddhist) -- non-axial?
get_match(53) # swaminarayan Sampdraday -- Hindu
get_match(46) # Nechung (Tibet Oracle), Dasara (Hindu), Shaiva (Hindu), Jain Digambara (Jainism) 
get_match(91) # tamil saiva (Shaivism), ladakhi buddhism

# top side (Indians) -- non-axial? 
get_match(93) # Peyote (Native American) -- Indians
get_match(145) # Timbira (Canela) -- Indians
get_match(60) # pythagoreanism -- ...
get_match(85) # Manus -- Christian/Indigenous mix

# left side 



### actually, we should scale AFTERWARDS ###
### i.e. maximum node should have same size, and maximum edge should have same 
### thickness --- or the total WEIGHT in the plot should be the same 

### NB: might want to select preferentially the "clean" configs
### rather than just maximum likelihood 

# backtrack some of the civs 
## just check a sample here 
### biggest difference
def scale_column(d, column): 
    col_lst = (d[column]-d[column].min())/(d[column].max()-d[column].min())
    return col_lst 

node_max_diff = node_attr[['node_id', 'p_val_x', 'config_sum']]
node_max_diff['p_scale'] = scale_column(node_max_diff, 'p_val_x')
node_max_diff['config_scale'] = scale_column(node_max_diff, 'config_sum')
node_max_diff = node_max_diff.assign(diff = lambda x: x['p_scale']-x['config_scale'])

large_data = node_max_diff.sort_values('diff', ascending=True).head(5)[['node_id']] 
large_model = node_max_diff.sort_values('diff', ascending=False).head(5)[['node_id']]

large_data = d_configurations.merge(large_data, on = 'node_id', how = 'inner')
large_model = d_configurations.merge(large_model, on = 'node_id', how = 'inner')

large_model.sort_values('node_id', ascending=True)
large_data.sort_values('node_id', ascending=True)

## check those with duplicate vales 
maxL_configs = d_configurations[d_configurations['max_likelihood'] == 'yes']
node_ml_duplicates = maxL_configs.groupby('node_id').size().reset_index(name = 'count')
node_ml_duplicates = node_ml_duplicates[node_ml_duplicates['count'] > 1]
node_ml_duplicates = maxL_configs.merge(node_ml_duplicates, on = 'node_id', how = 'inner')
node_ml_duplicates = node_ml_duplicates.sort_values(['node_id', 'entry_weight'], ascending = [True, False])
node_ml_duplicates = node_ml_duplicates[['node_id', 'entry_id', 'entry_name', 'entry_weight']]

node_ml_duplicates[node_ml_duplicates['node_id'] == 0]
node_ml_duplicates.iloc[11:20]
d_configurations


# only states with maximum likelihood 
## this does not do a lot 
## should probably just remove this 
G_maxL = G.copy()
for node in G_maxL.nodes():
    maxL = G_maxL.nodes[node]['maximum_likelihood'] 
    if maxL == 'no':
        G_maxL.nodes[node]['config_sum'] = 0
    else: 
        pass 

G_maxL = edge_strength(G_maxL, 'config_sum')
edgelst_maxL, edgew_maxL = edge_information(G_maxL, 'pmass_mult', 'weight', 0.2)
nodelst_maxL, nodesize_maxL = node_information(G_maxL, 'config_sum', 10)

fig, ax = plt.subplots(1, 2, facecolor = 'w', figsize = (10, 10), dpi = 500)
plt.axis('off')
draw_network(G_full, pos, 'Blues', 0.6, nodelst_full, nodesize_full, edgelst_full, edgew_full, 0)
draw_network(G_maxL, pos, 'Blues', 0.6, nodelst_maxL, nodesize_maxL, edgelst_maxL, edgew_maxL, 1)
plt.show();


# different ways of getting POS 
p_ind = d_ind['p_ind'].tolist()
top_states = allstates[p_ind]
h_distances = hamming_distance(top_states) 
hamming_full = hamming_edges(n_top_states, h_distances)

#### different layouts (position) #####
#### only the one distance pulls them appart properly ####
## one distance 
hamming_one_distance = hamming_full[hamming_full['hamming'] == 1]
hamming_one_distance['weight'] = hamming_one_distance['hamming']

## maximum value scaling
hamming_maxval = hamming_full['hamming'].max()
hamming_max_scaling = hamming_full.assign(weight = lambda x: hamming_maxval-x['hamming'])

## reciprocal scaling
hamming_reciprocal = hamming_full.assign(weight = lambda x: 1/x['hamming'])

## get positions for all of these 
position_lst = []
for df in [hamming_one_distance, hamming_max_scaling, hamming_reciprocal]: 
    G = nx.from_pandas_edgelist(df, 'node_x', 'node_y', 'weight')
    position = nx.nx_agraph.graphviz_layout(G, prog = 'fdp')
    position_lst.append(position)
pos_one_distance = position_lst[0]
pos_max_scaling = position_lst[1]
pos_reciprocal = position_lst[2]