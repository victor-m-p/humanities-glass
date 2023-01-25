# COGSCI23
import pandas as pd 
import numpy as np 
import configuration as cn 
from tqdm import tqdm 

# load documents
entry_maxlikelihood = pd.read_csv('../data/analysis/entry_maxlikelihood.csv')
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
question_reference = pd.read_csv('../data/analysis/question_reference.csv')

# generate all states
n_nodes = 20
from fun import bin_states 
configurations = bin_states(n_nodes) 

# get all unique configurations 
unique_configurations = entry_maxlikelihood['config_id'].unique().tolist()

# find id of neighbors
def get_neighbor_idx(configuration, indices, configurations):
    neighbor_idx_above_threshold = []
    for i in indices: 
        flip = configuration.flip_index(i)
        flipid = np.array([np.where((configurations == flip).all(1))[0][0]])[0]
        neighbor_idx_above_threshold.append(flipid)
    return neighbor_idx_above_threshold 

# move on to 
max_timestep = 100
threshold = 0.5 
sample_list = [] # (timestep, focal_idx, neighbor_idx, probability)
focal_idx_list = [unique_configurations[0]]

for t in range(max_timestep):
    # get all of our moves 
    neighbor_idx_total = []
    for focal_idx in focal_idx_list: 
        ConfObj = cn.Configuration(focal_idx,
                                   configurations,
                                   configuration_probabilities)
        p_move = ConfObj.p_move(configurations,
                                configuration_probabilities,
                                summary = False)
        # find indices and values 
        indices = np.argwhere(p_move > threshold)
        values = p_move[indices]
        # if there are indices then log, otherwise pass 
        if len(indices) != 0: 
            neighbor_idx_list = get_neighbor_idx(ConfObj, indices,
                                                 configurations)
            neighbor_idx_total.extend(neighbor_idx_list)
            # we log information
            for neighbor_idx, value in zip(neighbor_idx_list, values): 
                sample_list.append((t, focal_idx, neighbor_idx, value[0]))
    # break if there were no moves on a timestep 
    if not neighbor_idx_total: 
        break 
    else: 
        focal_idx_list = neighbor_idx_total 

sample_list

