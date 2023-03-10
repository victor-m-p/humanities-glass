'''
Helper functions for the analysis of DRH data.
VMP 2023-02-06: refactored with chatGPT and docstrings. 
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
    """correlation calculation in Ising equation

    Args:
        X (np.Array): terms inside logs
        coeffs (np.Array, optional): factors in front of exponential. Defaults to None.

    Returns:
        float: sum of exponentials
    """
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
    """return probabilities for 2**h states

    Args:
        h (np.Array): local fields
        J (np.Array): pairwise couplings. 

    Returns:
        np.Array: probabilities for all configurations
    """
    n_nodes = len(h)
    hJ = np.concatenate((h, J))
    h_combinations = np.array(list(itertools.product([1, -1], repeat = n_nodes)))
    J_combinations = np.array([list(itertools.combinations(i, 2)) for i in h_combinations])
    J_combinations = np.add.reduce(J_combinations, 2)
    J_combinations[J_combinations != 0] = 1
    J_combinations[J_combinations == 0] = -1
    condition_arr = np.concatenate((h_combinations, J_combinations), axis = 1)
    flipped_arr = hJ * condition_arr
    summed_arr = np.sum(flipped_arr, axis = 1)
    logsumexp_arr = fast_logsumexp(summed_arr)[0]
    Pout = np.exp(summed_arr - logsumexp_arr)
    return Pout[::-1]

def bin_states(n, sym=True):
    """generate 2**n possible configurations

    Args:
        n (int): number of questions (features)
        sym (bool, optional): symmetric system. Defaults to True.

    Returns:
        np.Array: 2**n configurations 
    """
    v = np.array([list(np.binary_repr(i, width=n)) for i in range(2**n)]).astype(int)
    if sym is False:
        return v
    return v*2-1

# '''https://stackoverflow.com/questions/42752610/python-how-to-generate-the-pairwise-hamming-distance-matrix'''
def hamming_distance(X):
    """Calculate Hamming distance

    Args:
        X (np.Array): Array of binary values (rows = configurations, columns = binary answers)

    Returns:
        np.Array: hamming distance (rows * rows)
    """
    return (X[:, None, :] != X).sum(2)

def top_n_idx(N, p, ind_colname='config_index', prob_colname='config_prob'):
    """get the most probable N states

    Args:
        N (int): number of configurations wanted
        p (np.Array): array of probabilities for configurations
        ind_colname (str, optional): desired column name for index column. Defaults to 'config_index'.
        val_colname (str, optional): desired column name for probability column. Defaults to 'config_prob'.

    Returns:
        pd.DataFrame: Dataframe with most probable N states, their index and probability
    """
    N = N+1
    val_cutoff = np.sort(p)[-N]
    p_ind = np.argwhere(p > val_cutoff).flatten()
    p_val = p[p_ind]
    data_out = pd.DataFrame({ind_colname: p_ind, prob_colname: p_val}).nlargest(N, prob_colname)
    return data_out.reset_index(drop = True)

def edge_information(Graph, weight_attribute, filter_attribute, scaling = 1): 
    """Return list of edges and list of edge weights, both sorted by edge weights (filtered, scaled)

    Args:
        Graph (nx.Graph): networkx graph object with weight_attribute and filter_attribute
        weight_attribute (str): weight attribute (could be other attribute, but should be numeric)
        filter_attribute (str): filter attribute (e.g. only hamming distance == 1).
        scaling (numeric): scaling of weights (for visualization purposes). Defaults to 1 (not scaled).

    Returns:
        lists: list of edges, list of edge weights. 
    """
    ## get edge attributes
    edge_weight = nx.get_edge_attributes(Graph, weight_attribute)
    edge_hdist = nx.get_edge_attributes(Graph, filter_attribute)

    ## sort edge weights by value
    edge_weights_sorted = sorted(edge_weight.items(), key=lambda x: x[1])
    edge_weights_filtered = [(k, v) for k, v in edge_weights_sorted if edge_hdist[k] == 1]
    
    # scale edge weights
    edge_weights_scaled = [(k, v * scaling) for k, v in edge_weights_filtered]
    
    # return edge list and scaled weights
    edge_list = [k for k, _ in edge_weights_scaled]
    edge_weights = [v for _, v in edge_weights_scaled]
    
    return edge_list, edge_weights

def hamming_edges(n_top_states, h_distances):
    idx = [f'hamming{x}' for x in range(n_top_states)]
    d = pd.DataFrame(h_distances, columns = idx)
    d['node_x'] = d.index
    d = pd.wide_to_long(d, stubnames = "hamming", i = 'node_x', j = 'node_y').reset_index()
    d = d[d['node_x'] != d['node_y']] # remove diagonal 
    d = d.drop_duplicates() # remove duplicates
    return d 

def hamming_edges(N, H_distances):
    """Get edgelist with hamming distance for the top N states

    Args:
        N (int): Number of configurations
        H_distances (np.Array): Array of hamming distances (shape N * N)

    Returns:
        _type_: _description_
    """
    col_names = [f'hamming{x}' for x in range(N)]
    df = pd.DataFrame(H_distances, columns=col_names)
    df['node_x'] = df.index
    df = pd.wide_to_long(df, stubnames="hamming", i='node_x', j='node_y').reset_index()
    df = df[df['node_x'] != df['node_y']]
    df.drop_duplicates(inplace=True)
    return df

def edge_strength(G, nodestrength):
    """Add multiplicative and additive edge strength based on node attribute

    Args:
        G (nx.Graph): networkx graph object
        nodestrength (str): node attribute (numeric)

    Returns:
        nx.Graph: New graph with added edge attributes
    """
    Gcopy = G.copy()
    for edge_x, edge_y in Gcopy.edges():
        pmass_x = Gcopy.nodes[edge_x][nodestrength]
        pmass_y = Gcopy.nodes[edge_y][nodestrength]
        Gcopy.edges[(edge_x, edge_y)].update({
            'pmass_mult': pmass_x * pmass_y,
            'pmass_add': pmass_x + pmass_y
        })
    return Gcopy

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

def sort_edge_attributes(Graph, weight_attribute, filter_attribute, scaling = 1): 
    """Return list of edges and list of edge weights, both sorted by edge weights (filtered, scaled)

    Args:
        Graph (nx.Graph): networkx graph object with weight_attribute and filter_attribute
        weight_attribute (str): weight attribute (could be other attribute, but should be numeric)
        filter_attribute (str): filter attribute (e.g. only hamming distance == 1).
        scaling (numeric): scaling of weights (for visualization purposes). Defaults to 1 (not scaled).

    Returns:
        lists: list of edges, list of edge weights. 
    """
    ## get edge attributes
    edge_weight = nx.get_edge_attributes(Graph, weight_attribute)
    edge_hdist = nx.get_edge_attributes(Graph, filter_attribute)

    ## sort edge weights by value
    edge_weights_sorted = sorted(edge_weight.items(), key=lambda x: x[1])
    edge_weights_filtered = [(k, v) for k, v in edge_weights_sorted if edge_hdist[k] == 1]
    
    # scale edge weights
    edge_weights_scaled = [(k, v * scaling) for k, v in edge_weights_filtered]
    
    # return edge list and scaled weights
    edge_list = [k for k, _ in edge_weights_scaled]
    edge_weights = [v for _, v in edge_weights_scaled]
    
    return edge_list, edge_weights

def sort_node_attributes(Graph, sorting_attribute, value_attribute):
    """Sort nodes based on attribute and return sorted node list and value list

    Args:
        Graph (nx.Graph): networkx graph object
        sorting_attribute (str): string containing sorting attribute
        value_attribute (str): string containing value attribute

    Returns:
        lst: list of sorted nodes and values
    """
    sorting_attr = nx.get_node_attributes(Graph, sorting_attribute)
    nodelist_sorted = [k for k, v in sorted(sorting_attr.items(), key=lambda item: item[1])]
    value_attr = nx.get_node_attributes(Graph, value_attribute)
    value_sorted = [v for k, v in sorted(value_attr.items(), key=lambda pair: nodelist_sorted.index(pair[0]))]
    return nodelist_sorted, value_sorted