import numpy as np 
from fun import top_n_idx, hamming_distance
import pandas as pd 
from sklearn.manifold import MDS

# setup
n_rows, n_nan, n_nodes, seed = 455, 5, 20, 254

# load stuff
p = np.loadtxt(f'../data/analysis/p_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
allstates = np.loadtxt(f'../data/analysis/allstates_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')
p_datastates = d_likelihood['p_ind'].unique().tolist()

# run 
for n_top_states in [5, 10, 20]: 
    
    d_ind = top_n_idx(n_top_states, p, 'p_ind', 'p_val') 
    p_ind = d_ind['p_ind'].tolist()
    p_val = d_ind['p_val'].tolist()
    top_states = allstates[p_ind]
    
    # loop
    n_nearest = 2
    lst_dfs = []
    for i, values in enumerate(zip(p_ind, p_val, top_states)): 
        idx_top, p_top, val_top = values
        lst_neighbors = []
        for idx_datastate in p_datastates:
            val_datastate = allstates[idx_datastate]
            h_dist = np.count_nonzero(val_top!=val_datastate)
            if h_dist <= n_nearest and idx_top != idx_datastate: 
                lst_neighbors.append((i, idx_top, idx_datastate, p_top, h_dist))
            else: 
                pass
        df_neighbor = pd.DataFrame(
            lst_neighbors, 
            columns = ['node_id', 'p_ind_focal', 'p_ind', 'config_w', 'hamming_neighbor'])
        lst_dfs.append(df_neighbor)
    df_hamming = pd.concat(lst_dfs)
    df_hamming.to_csv(f'../data/analysis/hamming_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}_ntop_{n_top_states}.csv', index = False)
    d_ind.to_csv(f'../data/analysis/d_index_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}_ntop_{n_top_states}.csv', index = False)
    
    ## MDS
    for n_dimensions in [2, 3]: 
        h_distances = hamming_distance(top_states) 
        mds = MDS(
            n_components = n_dimensions,
            n_init = 10, 
            max_iter = 2000, 
            eps=1e-8, 
            random_state=seed,
            dissimilarity='precomputed',
            n_jobs=-1 
        )
        mds_fit = mds.fit(h_distances) 
        embedding = mds_fit.embedding_ # mds.fit_transform(h_distances)
        stress = mds_fit.stress_ 
        kruskal = np.sqrt(stress / (0.5 * np.sum(h_distances**2)))
        np.savetxt(f'../data/analysis/pos_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}_ntop_{n_top_states}_ndim_{n_dimensions}_kruskal_{kruskal}.txt', embedding)