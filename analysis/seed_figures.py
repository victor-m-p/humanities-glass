#### sparta ####
#spartan_node_id = 35
n_nearest = 2
#d_spartan = d_max_weight[d_max_weight['node_id'] == spartan_node_id]
#spartan_idx = d_spartan['p_ind'].values[0]
spartan_idx = 769927 # roman imperial
spartan_main = get_n_neighbors(n_nearest, spartan_idx, allstates, p)

## sample the 150 top ones 
n_top_states = 99
spartan_cutoff = spartan_main.sort_values('prob_neighbor', ascending=False).head(n_top_states)
spartan_neighbor = spartan_cutoff[['idx_neighbor', 'prob_neighbor']]
spartan_neighbor = spartan_neighbor.rename(columns = {'idx_neighbor': 'p_ind',
                                                  'prob_neighbor': 'p_raw'})
spartan_focal = spartan_cutoff[['idx_focal', 'prob_focal']].drop_duplicates()
spartan_focal = spartan_focal.rename(columns = {'idx_focal': 'p_ind',
                                                'prob_focal': 'p_raw'})
sparta_ind = pd.concat([spartan_focal, spartan_neighbor])
sparta_ind = sparta_ind.reset_index(drop=True)
sparta_ind['node_id'] = sparta_ind.index

## now it is just the fucking pipeline again. 
sparta_overlap = datastate_information(d_likelihood, nodes_reference, sparta_ind) # 305
sparta_datastate_weight = datastate_weight(sparta_overlap) # 114
sparta_max_weight = maximum_weight(sparta_overlap, sparta_datastate_weight)
sparta_attr = merge_node_attributes(sparta_max_weight, sparta_ind)

## add hamming distance
spartan_hamming_neighbor = spartan_cutoff[['idx_neighbor', 'hamming']]
spartan_hamming_neighbor = spartan_hamming_neighbor.rename(columns = {'idx_neighbor': 'p_ind'})
spartan_hamming_focal = pd.DataFrame([(spartan_idx, 0)], columns = ['p_ind', 'hamming'])
spartan_hamming = pd.concat([spartan_hamming_focal, spartan_hamming_neighbor])
sparta_attr = sparta_attr.merge(spartan_hamming, on = 'p_ind', how = 'inner')
sparta_attr_dict = sparta_attr.to_dict('index')

# hamming distance
p_ind = sparta_ind['p_ind'].tolist()
top_states = allstates[p_ind]
h_distances = hamming_distance(top_states) 
h_distances = hamming_edges(n_top_states+1, h_distances)
h_distances = h_distances[h_distances['hamming'] == 1]

# create network
G = nx.from_pandas_edgelist(h_distances,
                            'node_x',
                            'node_y',
                            'hamming')

pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")

# add all node information
node_attr_dict
for idx, val in sparta_attr_dict.items(): 
    for attr in val: 
        idx = val['node_id']
        G.nodes[idx][attr] = val[attr]
        
# process 
G_full = edge_strength(G, 'p_raw') 
edgelst_full, edgew_full = edge_information(G_full, 'pmass_mult', 'hamming', 30000)
nodelst_full, nodesize_full = node_information(G_full, 'p_raw', 5000)

# color by spartan vs. other
color_lst = []
for node in nodelst_full: 
    hamming_dist = sparta_attr_dict.get(node)['hamming']
    color_lst.append(hamming_dist)
    
######### main plot ###########
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") # reverse code this

## slight manual tweak
pos_mov = pos.copy()
x, y = pos_mov[1]
pos_mov[1] = (x, y-10)

## now the plot 
nx.draw_networkx_nodes(G_full, pos_mov, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G_full, pos_mov, alpha = 0.7,
                       width = [x*3 for x in edgew_full],
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
plt.savefig('../fig/seed_RomanImpCult.pdf')

######### reference plot ##########
labeldict = {}
for node in nodelst_full:
    node_id = G_full.nodes[node]['node_id']
    labeldict[node] = node_id

fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") # reverse code this
nx.draw_networkx_nodes(G_full, pos, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G_full, pos, alpha = 0.7,
                       width = edgew_full,
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
label_options = {"ec": "k", "fc": "white", "alpha": 0.1}
nx.draw_networkx_labels(G_full, pos, font_size = 8, labels = labeldict, bbox = label_options)
plt.savefig('../fig/seed_RomanImpCult_reference.pdf')

get_match(sparta_max_weight, 2) # Mesopotamia, Thessalians
get_match(sparta_max_weight, 1) # Egypt
get_match(sparta_max_weight, 6) # Achaemenid 
get_match(sparta_max_weight, 3) # Luguru
get_match(sparta_max_weight, 4) # Pontifex Maximus
get_match(sparta_max_weight, 5) # Old Assyrian
get_match(sparta_max_weight, 7) # Archaic Spartan cults

## table for the top states ##
n_labels = 10
sparta_ind_raw = sparta_max_weight[['p_ind', 'p_raw']].drop_duplicates()
sparta_top_configs = sparta_ind_raw.sort_values('p_raw', ascending = False).head(n_labels)
sparta_top_configs = sparta_max_weight.merge(sparta_top_configs, on = ['p_ind', 'p_raw'], how = 'inner')
sparta_max_weight[sparta_max_weight['p_ind'] == spartan_idx]

######## free methodists
spartan_node_id = 18
n_nearest = 2
d_spartan = d_max_weight[d_max_weight['node_id'] == spartan_node_id]
spartan_idx = d_spartan['p_ind'].values[0]
#spartan_idx = 769927 # roman imperial
spartan_main = get_n_neighbors(n_nearest, spartan_idx, allstates, p)

## sample the 150 top ones 
n_top_states = 99
spartan_cutoff = spartan_main.sort_values('prob_neighbor', ascending=False).head(n_top_states)
spartan_neighbor = spartan_cutoff[['idx_neighbor', 'prob_neighbor']]
spartan_neighbor = spartan_neighbor.rename(columns = {'idx_neighbor': 'p_ind',
                                                  'prob_neighbor': 'p_raw'})
spartan_focal = spartan_cutoff[['idx_focal', 'prob_focal']].drop_duplicates()
spartan_focal = spartan_focal.rename(columns = {'idx_focal': 'p_ind',
                                                'prob_focal': 'p_raw'})
sparta_ind = pd.concat([spartan_focal, spartan_neighbor])
sparta_ind = sparta_ind.reset_index(drop=True)
sparta_ind['node_id'] = sparta_ind.index

## now it is just the fucking pipeline again. 
sparta_overlap = datastate_information(d_likelihood, nodes_reference, sparta_ind) # 305
sparta_datastate_weight = datastate_weight(sparta_overlap) # 114
sparta_max_weight = maximum_weight(sparta_overlap, sparta_datastate_weight)
sparta_attr = merge_node_attributes(sparta_max_weight, sparta_ind)

## add hamming distance
spartan_hamming_neighbor = spartan_cutoff[['idx_neighbor', 'hamming']]
spartan_hamming_neighbor = spartan_hamming_neighbor.rename(columns = {'idx_neighbor': 'p_ind'})
spartan_hamming_focal = pd.DataFrame([(spartan_idx, 0)], columns = ['p_ind', 'hamming'])
spartan_hamming = pd.concat([spartan_hamming_focal, spartan_hamming_neighbor])
sparta_attr = sparta_attr.merge(spartan_hamming, on = 'p_ind', how = 'inner')
sparta_attr_dict = sparta_attr.to_dict('index')

# hamming distance
p_ind = sparta_ind['p_ind'].tolist()
top_states = allstates[p_ind]
h_distances = hamming_distance(top_states) 
h_distances = hamming_edges(n_top_states+1, h_distances)
h_distances = h_distances[h_distances['hamming'] == 1]

# create network
G = nx.from_pandas_edgelist(h_distances,
                            'node_x',
                            'node_y',
                            'hamming')

pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")

# add all node information
node_attr_dict
for idx, val in sparta_attr_dict.items(): 
    for attr in val: 
        idx = val['node_id']
        G.nodes[idx][attr] = val[attr]
        
# process 
G_full = edge_strength(G, 'p_raw') 
edgelst_full, edgew_full = edge_information(G_full, 'pmass_mult', 'hamming', 30000)
nodelst_full, nodesize_full = node_information(G_full, 'p_raw', 5000)

# color by spartan vs. other
color_lst = []
for node in nodelst_full: 
    hamming_dist = sparta_attr_dict.get(node)['hamming']
    color_lst.append(hamming_dist)
    
######### main plot ###########
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") # reverse code this

## slight manual tweak

## now the plot 
nx.draw_networkx_nodes(G_full, pos_mov, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G_full, pos_mov, alpha = 0.7,
                       width = [x*5 for x in edgew_full],
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
plt.savefig('../fig/seed_FreeMethChurch.pdf')

######### reference plot ##########
labeldict = {}
for node in nodelst_full:
    node_id = G_full.nodes[node]['node_id']
    labeldict[node] = node_id

fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") # reverse code this
nx.draw_networkx_nodes(G_full, pos, 
                        nodelist = nodelst_full,
                        node_size = [x*2 for x in nodesize_full], 
                        node_color = [3-x for x in color_lst],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
rgba = rgb2hex(cmap(0.9))
nx.draw_networkx_edges(G_full, pos, alpha = 0.7,
                       width = edgew_full,
                       edgelist = edgelst_full,
                       edge_color = rgba
                       )
label_options = {"ec": "k", "fc": "white", "alpha": 0.1}
nx.draw_networkx_labels(G_full, pos, font_size = 8, labels = labeldict, bbox = label_options)
plt.savefig('../fig/seed_FreeMethChurch_reference.pdf')

get_match(sparta_max_weight, 0) # Free methodist
get_match(sparta_max_weight, 1) # lots of shit
get_match(sparta_max_weight, 2) # Southern Baptists
get_match(sparta_max_weight, 3) # Messalians
get_match(sparta_max_weight, 4) # Nothing (empty)
get_match(sparta_max_weight, 5) # Sachchai
get_match(sparta_max_weight, 9) # Pauline Christianity (45-60 CE)
get_match(sparta_max_weight, 13) # Nothing (empty)

## which traits would have to be changed ##
# ...

##### features ######
## we need to scale this differently: 
## (1) we need to weigh by probability of configuration
## (2) take the weighted mean configuration for each 
## (3) compare against the weighted mean of the rest of the states

sref = pd.read_csv('../data/analysis/sref_nrows_455_maxna_5_nodes_20.csv')
question_ids = sref['related_q_id'].to_list()

def state_agreement(d, config_lst): 
    
    # subset states 
    p_ind_uniq = d[d['node_id'].isin(config_lst)]
    p_ind_uniq = p_ind_uniq['p_ind'].unique()
    p_ind_uniq = list(p_ind_uniq)

    # get the configurations
    d_conf = allstates[p_ind_uniq]

    # to dataframe 
    d_mat = pd.DataFrame(d_conf, columns = question_ids)
    d_mat['p_ind'] = p_ind_uniq
    d_mat = pd.melt(d_mat, id_vars = 'p_ind', value_vars = question_ids, var_name = 'related_q_id')
    d_mat = d_mat.replace({'value': {-1: 0}})
    d_mat = d_mat.groupby('related_q_id')['value'].mean().reset_index(name = 'mean_val')

    # merge back in question names
    d_interpret = d_mat.merge(sref, on = 'related_q_id', how = 'inner')
    d_interpret = d_interpret.sort_values('mean_val')

    # return 
    return d_interpret

# run on the big communities
pd.set_option('display.max_colwidth', None)

def disagreement_across(d):
    d_std = d.groupby('related_q')['mean_val'].std().reset_index(name = 'standard_deviation')
    d_mean = d.groupby('related_q')['mean_val'].mean().reset_index(name = 'mean_across')
    d_final = d_std.merge(d_mean, on = 'related_q', how = 'inner')
    d_final = d_final.sort_values('standard_deviation', ascending=False)
    return d_final 

##### labels #####
# for this we should do the top 3 (or something) most highly
# weighted configurations per community and then take the corresponding
# religions (preferrably unique, maybe only considering unique). 
def get_match(d, n):
    dm = d[d['node_id'] == n][['entry_name']]
    print(dm.head(10))

get_match(18) # Free Methodist Church
get_match(27) # Roman imperial cult
get_match(35) # Archaic Spartan cult. 

########### old shit ###########
def state_agreement(d, config_lst): 
    
    # subset states 
    p_ind_uniq = d[d['node_id'].isin(config_lst)]
    p_ind_uniq = p_ind_uniq['p_ind'].unique()
    p_ind_uniq = list(p_ind_uniq)

    # get the configurations
    d_conf = allstates[p_ind_uniq]

    # to dataframe 
    d_mat = pd.DataFrame(d_conf, columns = question_ids)
    d_mat['p_ind'] = p_ind_uniq
    d_mat = pd.melt(d_mat, id_vars = 'p_ind', value_vars = question_ids, var_name = 'related_q_id')
    d_mat = d_mat.replace({'value': {-1: 0}})
    d_mat = d_mat.groupby('related_q_id')['value'].mean().reset_index(name = 'mean_val')

    # merge back in question names
    d_interpret = d_mat.merge(sref, on = 'related_q_id', how = 'inner')
    d_interpret = d_interpret.sort_values('mean_val')

    # return 
    return d_interpret

# run on the big communities
pd.set_option('display.max_colwidth', None)

def disagreement_across(d):
    d_std = d.groupby('related_q')['mean_val'].std().reset_index(name = 'standard_deviation')
    d_mean = d.groupby('related_q')['mean_val'].mean().reset_index(name = 'mean_across')
    d_final = d_std.merge(d_mean, on = 'related_q', how = 'inner')
    d_final = d_final.sort_values('standard_deviation', ascending=False)
    return d_final 

