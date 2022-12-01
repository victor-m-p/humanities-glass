import numpy as np 
import pandas as pd 

# helper functions 
def flip_bit(state): 
    return -1 if state == 1 else 1

def idx_row_overlap(A, B): 
    return np.where((A == B).all(1))[0][0]

# 1. combine information 
df_likelihood = pd.read_csv('../data/analysis/d_likelihood_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.csv')

## this is general information that would be good to have 
df_sample_reference = pd.read_csv('../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv')
df_likelihood = df_likelihood.merge(df_sample_reference, on = 'entry_id', how = 'inner')

## this is general information that would be good to have 
df_entry_count = df_likelihood.groupby('entry_id').size().reset_index(name = 'entry_count')
df_entry_count = df_entry_count.assign(entry_weight = lambda x: 1/x['entry_count'])
df_likelihood = df_likelihood.merge(df_entry_count, on = 'entry_id', how = 'inner')

# 2. get only complete records 
df_complete_records = df_likelihood[df_likelihood['entry_count'] == 1]

# 3. couple this to actual states in the data with p_ind 
config_prob = np.loadtxt('../data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt')
allstates = np.loadtxt('../data/analysis/allstates_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt').astype(int)


# 4. loop over bitflips 
entry_id = 174 
p_ind = 370564

base_state = allstates[p_ind]
base_prob = config_prob[p_ind]

# probability for all neighboring states 
bit_flip_probs = []
for i in range(len(base_state)): 
    # find idx and probability of flipped state
    base_state_copy = base_state 
    base_state_copy[i] = flip_bit(base_state[i])
    base_state_copy_idx = idx_row_overlap(allstates, base_state_copy)
    base_state_copy_prob = config_prob[base_state_copy_idx]
    
    # log data
    bit_flip_probs.append(base_state_copy_prob)

bit_flip_probs
