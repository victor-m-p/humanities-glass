import numpy as np
import itertools 
import pandas as pd 

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
def top_n_idx(n, p): # fix this
    val_cutoff = np.sort(p)[::-1][n]
    p_ind = [i for i, v in enumerate(p) if v > val_cutoff]
    p_vals = p[p > val_cutoff]
    return p_ind, p_vals
