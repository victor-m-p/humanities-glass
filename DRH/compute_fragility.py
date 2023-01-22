# COGSCI23
import pandas as pd 
import numpy as np 
import configuration as cn 
from tqdm import tqdm 

# load documents
entry_configuration_master = pd.read_csv('../data/analysis/entry_configuration_master.csv')
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
question_reference = pd.read_csv('../data/analysis/question_reference.csv')

# generate all states
n_nodes = 20
from fun import bin_states 
configurations = bin_states(n_nodes) 

# get all unique configurations 
unique_configurations = entry_configuration_master['config_id'].unique().tolist()

# around 15 minutes on local computer 
transition_list = []
for configuration in tqdm(unique_configurations): 
    conf = cn.Configuration(configuration,
                            configurations, 
                            configuration_probabilities)
    p_move = conf.p_move(configurations, 
                         configuration_probabilities,
                         summary = False)
    p_move = sorted(p_move, reverse = True)
    for n_fixed in range(20): 
        p_move_n = p_move[n_fixed:]
        p_mean_n = np.mean(p_move_n)
        transition_list.append((configuration, n_fixed, p_mean_n))

# save this 
d = pd.DataFrame(transition_list, columns = ['config_id', 'n_fixed_traits', 'prob_move'])
d.to_csv('../data/COGSCI23/fragility_observed.csv', index = False)