import numpy as np 
from fun import p_dist, bin_states, top_n_idx, hamming_distance
import pandas as pd 
from sklearn.manifold import MDS

# setup
n_nodes = 20
n_nan = 5

# important reference for now (only states that have these ids)
infile = f'../data/mdl_final/reference_with_entry_id_cleaned_nrows_455_maxna_{n_nan}.dat'
with open(infile) as f:
    reference = [x.strip() for x in f.readlines()]
reference = [int(x.split()[0]) for x in reference]
d_reference = pd.DataFrame({'entry_id': reference})
d_reference = d_reference.drop_duplicates()

# recreate the node reference document
nodes_reference = pd.read_csv('../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv')
nodes_reference = nodes_reference.merge(d_reference, on = 'entry_id', how = 'inner')
nodes_reference.to_csv(f'../data/analysis/nref_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.csv', index = False)
states_reference = pd.read_csv('../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv')
states_reference.to_csv(f'../data/analysis/sref_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.csv', index = False)

# collapse weighted rows to nan
d_main = pd.read_csv('../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv')
d_main = d_main.rename(columns = {'s': 'entry_id'})
d_main = d_main.merge(d_reference, on = 'entry_id', how = 'inner')
qcols = d_main.columns[1:-1]
d_unweighted = d_main.groupby('entry_id')[qcols].mean().reset_index().astype(int)
d_unweighted.to_csv(f'../data/analysis/d_collapsed_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.csv', index = False)
d_main.to_csv(f'../data/analysis/d_main_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.csv', index = False)

# calculate probability of all configurations based on parameters h, J.
params = np.loadtxt('../data/mdl_final/cleaned_nrows_455_maxna_5.dat_params.dat')
n_nodes = 20
nJ = int(n_nodes*(n_nodes-1)/2)
J = params[:nJ]
h = params[nJ:]
p = p_dist(h, J) # this takes some time (and should not be attempted with n_nodes > 20)
np.savetxt(f'../data/analysis/p_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.txt', p)

# allstates
allstates = bin_states(n_nodes) # takes a minute (do not attempt with n_nodes > 20)
np.savetxt(f'../data/analysis/allstates_nrows_455_maxna_{n_nan}_nodes_{n_nodes}.txt', allstates.astype(int), fmt='%i')

# MDS (obtain positions)
seed = 254
c_cutoff = 500
outname = '../data/analysis/pos_nrow_....'

p_ind, p_vals = top_n_idx(c_cutoff, p) 
top_states = allstates[p_ind]
h_distances = hamming_distance(top_states) 
mds = MDS(
    n_components = 2,
    n_init = 6, 
    max_iter = 1000, 
    eps=1e-7, 
    random_state=seed,
    dissimilarity='precomputed',
    n_jobs=-1 
)
mds_fit = mds.fit(h_distances) 
mds_fit.embedding_ # mds.fit_transform(h_distances)
mds_fit.stress_ 
np.savetxt(f'../data/analysis/pos_nrows_455_maxna_{n_nan}_nodes_{n_nodes}_cutoff_{c_cutoff}_seed_{seed}.txt', pos)


### test different fits ###
cutoff_lst = [100, 200, 300, 400, 500]
mds_stress = {}
mds_embed = {}
for cutoff in cutoff_lst: 
    p_ind, p_vals = top_n_idx(cutoff, p) 
    top_states = allstates[p_ind]
    h_distances = hamming_distance(top_states) 
    mds = MDS(
        n_components = 2,
        n_init = 6, 
        max_iter = 1000, 
        eps=1e-7, 
        random_state=seed,
        dissimilarity='precomputed',
        n_jobs=-1 
    )
    mds_fit = mds.fit(h_distances) 
    embedding = mds_fit.embedding_ # mds.fit_transform(h_distances)
    stress = mds_fit.stress_ 
    mds_stress[cutoff] = stress
    mds_embed[cutoff] = embedding

cutoff_lst = [5, 10, 20, 50, 100, 200, 300, 400, 500]
mds_stress = {}
mds_embed = {}
mds_stress1 = {}
for cutoff in cutoff_lst: 
    p_ind, p_vals = top_n_idx(cutoff, p) 
    top_states = allstates[p_ind]
    h_distances = hamming_distance(top_states) 
    mds = MDS(
        n_components = 2,
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
    mds_stress[cutoff] = stress
    mds_embed[cutoff] = embedding
    # https://stackoverflow.com/questions/36428205/stress-attribute-sklearn-manifold-mds-python
    mds_stress1[cutoff] = np.sqrt(stress / (0.5 * np.sum(h_distances**2)))

mds_stress1
cutoff_lst = [5, 10, 20, 50, 100, 200, 300, 400, 500]
mds_stress_3D = {}
mds_embed_3D = {}
mds_stress1_3D = {}
for cutoff in cutoff_lst: 
    p_ind, p_vals = top_n_idx(cutoff, p) 
    top_states = allstates[p_ind]
    h_distances = hamming_distance(top_states) 
    mds = MDS(
        n_components = 3,
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
    mds_stress_3D[cutoff] = stress
    mds_embed_3D[cutoff] = embedding
    # https://stackoverflow.com/questions/36428205/stress-attribute-sklearn-manifold-mds-python
    mds_stress1_3D[cutoff] = np.sqrt(stress / (0.5 * np.sum(h_distances**2)))

mds_stress1_3D

#### new MDS ####
## run for 2, 3 dimensions
## run for 5, 10 top states 
## for each of these states, find all neighbor probability mass
d_likelihood = pd.read_csv('../data/analysis/d_likelihood_nrows_455_maxna_5_nodes_20.csv')
p_datastates = d_likelihood['p_ind'].unique().tolist()


p_ind, p_vals = top_n_idx(5, p) 
top_states = allstates[p_ind]

d_likelihood = pd.read_csv('../data/analysis/d_likelihood_nrows_455_maxna_5_nodes_20.csv')
p_datastates = d_likelihood['p_ind'].unique().tolist()
n_nearest = 2
lst_dfs = []
for idx_top, val_top in zip(p_ind, top_states): 
    lst_neighbors = []
    for idx_datastate in p_datastates:
        val_datastate = allstates[idx_datastate]
        h_dist = np.count_nonzero(val_top!=val_datastate)
        if h_dist <= n_nearest and idx_top != idx_datastate: 
            lst_neighbors.append((idx_top, idx_datastate, h_dist))
        else: 
            pass
    df_neighbor = pd.DataFrame(
        lst_neighbors, 
        columns = ['p_ind_focal', 'p_ind', 'hamming_neighbor'])
    lst_dfs.append(df_neighbor)
df_hamming = pd.concat(lst_dfs)

h_distances = hamming_distance(top_states) 
mds = MDS(
    n_components = 3,
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
mds_stress_3D[cutoff] = stress
mds_embed_3D[cutoff] = embedding
# https://stackoverflow.com/questions/36428205/stress-attribute-sklearn-manifold-mds-python
mds_stress1_3D[cutoff] = np.sqrt(stress / (0.5 * np.sum(h_distances**2)))
