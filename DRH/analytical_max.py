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
def get_neighbor_idx(configuration, index, configurations):
    flip = configuration.flip_index(index)
    flipid = np.array([np.where((configurations == flip).all(1))[0][0]])[0]
    return flipid 

# move on to 
max_timestep = 100
threshold = 0.5 
sample_list = [] # (timestep, focal_idx, neighbor_idx, probability)

dataframe_list = []
for focal_idx in tqdm(unique_configurations): 
    original_idx = focal_idx
    sample_list = []
    for t in range(max_timestep):
        ConfObj = cn.Configuration(focal_idx,
                                configurations, 
                                configuration_probabilities)
        p_move = ConfObj.p_move(configurations, 
                                configuration_probabilities,
                                summary = False)
        # find indices and values 
        index = np.argmax(p_move)
        value = p_move[index]
        # if there are indices then log, otherwise pass 
        if value > threshold: 
            neighbor_idx = get_neighbor_idx(ConfObj,
                                            index,
                                            configurations)
            # we log information
            sample_list.append((t, focal_idx, neighbor_idx, value))
            focal_idx = neighbor_idx
        else: 
            break 
    sample_df = pd.DataFrame(sample_list, 
                             columns = ['timestep', 'config_from', 
                                        'config_to', 'weight'])
    sample_df.to_csv(f'../data/COGSCI23/max_attractor/idx{original_idx}.csv', index = False)
