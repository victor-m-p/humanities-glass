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
from tqdm import tqdm 

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

def transition_probabilities(configurations, configuration_probabilities,
                             idx_init, samples, sample_config_idx = False): 
    """transition probabilities for two neighbor questions

    Args:
        configurations (np.Array): all configurations (2**20)
        configuration_probabilities (np.Array): all configuration probabilities (2**20)
        idx_init (int): index (question number) for the first question
        samples (int): number of samples 
        sample_config_idx (list): optionally provide a specific list of indices
    
    Returns: Pd.Dataframe with transition probabilities
    
    NB: The two questions currently must be neighbors in the question list,
    e.g. have index 14, and 15 in the question list (in which case idx_init = 14). 

    NB: if specifying sample_config_idx note that this corresponds to the index
    of the row in 'restricted_configurations', not 'configurations'. 
    
    """
    # reproducibility
    np.random.seed(seed = 1)
    # configurations ignoring two indices (i.e. 2**18 rather than 2**20)
    restricted_configurations = np.delete(configurations, [idx_init, idx_init + 1], 1)
    restricted_configurations = np.unique(restricted_configurations, axis = 0)
    n_configurations = len(restricted_configurations)
    if not sample_config_idx: 
        sample_config_idx = np.random.choice(n_configurations,
                                             size = samples,
                                             replace = False)
    sample_configs = restricted_configurations[[sample_config_idx]]
    # set up labels 
    labels = range(4)
    # loop over the sample and calculate
    transition_probabilities = []
    for num, x in tqdm(enumerate(sample_configs)): 
        # get the configurations 
        conf_both = np.insert(x, idx_init, [1, 1])
        conf_none = np.insert(x, idx_init, [-1, -1])
        conf_first = np.insert(x, idx_init, [1, -1])
        conf_second = np.insert(x, idx_init, [-1, 1])
        # get the configuration idx 
        idx_both = np.where(np.all(configurations == conf_both, axis = 1))[0][0]
        idx_none = np.where(np.all(configurations == conf_none, axis = 1))[0][0]
        idx_first = np.where(np.all(configurations == conf_first, axis = 1))[0][0]
        idx_second = np.where(np.all(configurations == conf_second, axis = 1))[0][0]
        # get probabilities
        p_both = configuration_probabilities[idx_both]
        p_none = configuration_probabilities[idx_none]
        p_first = configuration_probabilities[idx_first]
        p_second = configuration_probabilities[idx_second]
        # gather in list 
        probabilities = [p_both, p_none, p_first, p_second]
        # put this together
        for p_focal, type_focal in zip(probabilities, labels): 
            if type_focal == labels[0] or type_focal == labels[1]: 
                p_neighbors = [probabilities[2], probabilities[3]]
                type_neighbors = [labels[2], labels[3]]
            else: 
                p_neighbors = [probabilities[0], probabilities[1]]
                type_neighbors = [labels[0], labels[1]]
            for p_neighbor, type_neighbor in zip(p_neighbors, type_neighbors): 
                flow = p_neighbor / (p_focal + sum(p_neighbors))
                transition_probabilities.append((num, type_focal, type_neighbor, flow))

    x = [(x, y, z) for _, x, y, z in transition_probabilities]
    df = pd.DataFrame(x, columns = ['type_from', 'type_to', 'probability'])
    df = df.groupby(['type_from', 'type_to'])['probability'].mean().reset_index(name = 'probability')
    return df 