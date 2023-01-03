import pandas as pd 
import numpy as np 
import configuration as cn 

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

# 
conf = cn.Configuration(unique_configurations[0], 
                        configurations, 
                        configuration_probabilities)

conf
