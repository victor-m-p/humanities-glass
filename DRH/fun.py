'''
VMP 2022-12-12: 
Helper functions for the analysis of DRH data.
'''

import numpy as np
import itertools 
import pandas as pd 
import networkx as nx 
import matplotlib.pyplot as plt 
from matplotlib.colors import rgb2hex

# taken from coniii enumerate
def fast_logsumexp(X, coeffs=None):
    Xmx = max(X)
    if coeffs is None:
        y = np.exp(X-Xmx).sum()
    else:
        y = np.exp(X-Xmx).dot(coeffs)

    if y<0:
        return np.log(np.abs(y))+Xmx, -1.
    return np.log(y)+Xmx, 1.

# still create J_combinations is slow for large number of nodes
def p_dist(h, J):
    # setup 
    n_nodes = len(h)
    n_rows = 2**n_nodes
    Pout = np.zeros((n_rows))
    
    ## hJ
    hJ = np.concatenate((h, J))
    
    ## put h, J into array
    parameter_arr = np.repeat(hJ[np.newaxis, :], n_rows, axis=0)

    ## True/False for h
    print('start h comb')
    h_combinations = np.array(list(itertools.product([1, -1], repeat = n_nodes)))
    
    print('start J comb') 
    ## True/False for J (most costly part is the line below)
    J_combinations = np.array([list(itertools.combinations(i, 2)) for i in h_combinations])
    J_combinations = np.add.reduce(J_combinations, 2) # if not == 0 then x == y
    J_combinations[J_combinations != 0] = 1
    J_combinations[J_combinations == 0] = -1
    
    # concatenate h, J
    condition_arr = np.concatenate((h_combinations, J_combinations), axis = 1) # what if this was just +1 and -1

    # multiply parameters with flips 
    flipped_arr = parameter_arr * condition_arr 
    
    # sum along axis 1
    summed_arr = np.sum(flipped_arr, axis = 1) 
    
    ## logsumexp
    print('start logsum')
    logsumexp_arr = fast_logsumexp(summed_arr)[0] # where is this function
    
    ## last step
    for num, ele in enumerate(list(summed_arr)):
        Pout[num] = np.exp(ele - logsumexp_arr)
    
    ## return stuff
    return Pout[::-1]

# taken from conii 
# but maybe this does not make sense now 
def bin_states(n, sym=True):
    v = np.array([list(np.binary_repr(i,width=n)) for i in range(2**n)]).astype(int)
    if sym is False:
        return v
    return v*2-1

# stackoverflow
def hamming_distance(X):
    '''https://stackoverflow.com/questions/42752610/python-how-to-generate-the-pairwise-hamming-distance-matrix'''
    return (X[:, None, :] != X).sum(2)

# 
def top_n_idx(n, p, ind_colname, val_colname): # fix this
    val_cutoff = np.sort(p)[::-1][n]
    p_ind = [i for i, v in enumerate(p) if v > val_cutoff]
    p_val = p[p > val_cutoff]
    d = pd.DataFrame({
        ind_colname: p_ind, 
        val_colname: p_val})
    d = d.sort_values(val_colname, ascending=False).reset_index(drop=True)
    return d


def draw_network(Graph, pos, cmap_name, alpha, nodelst, nodesize, nodecolor, edgelst, edgesize, axis, cmap_edge = 1): 
    cmap = plt.cm.get_cmap(cmap_name)
    nx.draw_networkx_nodes(Graph, pos, 
                           nodelist = nodelst,
                           node_size = nodesize, 
                           node_color = nodecolor,
                           linewidths = 0.5, edgecolors = 'black',
                           cmap = cmap,
                           ax = axis)
    rgba = rgb2hex(cmap(cmap_edge))
    nx.draw_networkx_edges(Graph, pos, width = edgesize, 
                        alpha = alpha, edgelist = edgelst,
                        edge_color = rgba,
                        #edge_color = edgesize,
                        #edge_cmap = cmap,
                        ax = axis)
    axis.set_axis_off()

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

## weight for configurations (proportional to data state weight) 
def datastate_weight(d_overlap): 
    d_entry_node = d_overlap[['entry_id', 'node_id']]
    d_datastate_weight = d_entry_node.groupby('entry_id').size().reset_index(name = 'entry_count')
    d_datastate_weight = d_datastate_weight.assign(entry_weight = lambda x: 1/x['entry_count'])
    d_datastate_weight = d_entry_node.merge(d_datastate_weight, on = 'entry_id', how = 'inner')
    d_datastate_weight = d_datastate_weight.groupby('node_id')['entry_weight'].sum().reset_index(name = 'datastate_sum')
    return d_datastate_weight 

def maximum_weight(d_overlap, d_datastate_weight): 
    d_max_weight = d_overlap.groupby('node_id')['p_norm'].max().reset_index(name = 'p_norm')
    d_max_weight = d_overlap.merge(d_max_weight, on = ['node_id', 'p_norm'], how = 'inner')
    d_max_weight = d_datastate_weight.merge(d_max_weight, on = 'node_id', how = 'inner')
    return d_max_weight

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

def weighted_average(df, values, weights):
    return sum(df[weights] * df[values]) / df[weights].sum()

def avg_bitstring(allstates, node_attr, question_ids, node_id_list, node_id_col, config_id_col, question_id_col, weight_col):
    focal_state = node_attr[node_attr[node_id_col].isin(node_id_list)]
    focal_uniq = focal_state[config_id_col].unique()
    focal_uniq = list(focal_uniq)
    focal_configs = allstates[focal_uniq]
    focal_mat = pd.DataFrame(focal_configs, columns = question_ids)
    focal_mat[config_id_col] = focal_uniq
    focal_mat = pd.melt(focal_mat, id_vars = config_id_col, value_vars = question_ids, var_name = question_id_col)
    # recode -1 to 0
    focal_mat = focal_mat.replace({'value': {-1: 0}})
    # weighting by probability
    focal_weights = focal_state[[config_id_col, weight_col]].drop_duplicates()
    focal_weights = focal_mat.merge(focal_weights, on = config_id_col, how = 'inner')
    focal_weights = focal_weights.groupby(question_id_col).apply(weighted_average, 'value', weight_col).reset_index(name = 'weighted_avg')
    return focal_weights