import numpy as np 
import pandas as pd 
from sim_fun import bin_states
import itertools
pd.set_option('display.max_colwidth', None)

# setup 
n_nodes, maxna = 20, 10
seed = 2
n_cutoff = 500
outpath = '../fig'

# read files 
p_file = '../data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt'
mat_file = '../data/clean/matrix_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.txt'
d_main = '../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
sref = '../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
nref = '../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
p = np.loadtxt(p_file)
d_main = pd.read_csv(d_main)
sref = pd.read_csv(sref)
nref = pd.read_csv(nref)
#datastates_weighted = np.loadtxt(mat_file)
#datastates = np.delete(datastates_weighted, n_nodes, 1)
#datastates_uniq = np.unique(datastates, axis = 0) # here we loose correspondence

# get all state configurations 
allstates = bin_states(n_nodes) # takes a minute (do not attempt with n_nodes > 20)

# get columns for indexing 
allcols = d_main.columns
qcols = allcols[1:-1]

# expanding nan data states 
def expand_nan(l): 
    N = len(l)
    lc = [[1, -1] if x == 0 else [x] for x in l]
    ### get back and check this 
    ### I think that we only need [p for p in itertools.product(*lc)]
    #states = np.array([p for c in itertools.combinations(lc, N) for p in itertools.product(*c)])
    states = np.array([p for p in itertools.product(*lc)])
    return states

# get probabilities
def get_probs(states, allstates, p): 
    state_ind = np.array(np.all((states[:,None,:]==allstates[None,:,:]), axis = -1).nonzero()).T.tolist()
    state_ref = [y for x, y in state_ind]
    probs = list(p[state_ref])
    probs_norm = [float(i)/sum(probs) for i in probs]
    return probs, probs_norm

# run over all cases (4 for now...)
#### creation of d_unweighted now in prep ####
d_unweighted = d_main.groupby('s')[qcols].mean().reset_index()
d_unweighted = d_unweighted.astype(int)
d_unweighted = d_unweighted.head(5)
l_mat, l_eid, l_p, l_pnorm = [], [], [], []
for s in d_unweighted['s'].unique(): 
    print(s)
    row = d_unweighted.loc[d_unweighted.s == s, qcols].values.tolist()
    lst = [item for sublist in row for item in sublist]
    stat = expand_nan(lst)
    probs, probs_norm = get_probs(stat, allstates, p)
    
    # get stuff out 
    rows, cols = stat.shape
    entry_id = np.repeat(s, rows)
    mat = np.hstack((entry_id[:, None], stat))
    l_mat.append(mat)
    l_eid.append(list(entry_id))
    l_p.append(probs)
    l_pnorm.append(probs_norm) 

# organize the values 
mat_stack = np.vstack(l_mat)
flat_eid = [item for sublist in l_eid for item in sublist]
flat_p = [item for sublist in l_p for item in sublist]
flat_pnorm = [item for sublist in l_pnorm for item in sublist]
df = pd.DataFrame({
    'entry_id': flat_eid,
    'p': flat_p,
    'p_norm': flat_pnorm})
df['index'] = df.index

## 36

# try to find e.g. maximum likelihood config for each
max_likelihood = df.groupby('entry_id')['p', 'p_norm', 'index'].max().reset_index()
max_likelihood
mat_stack[3][:] # maximum likelihood config for this record

# now ... try to translate this to Julia ... 