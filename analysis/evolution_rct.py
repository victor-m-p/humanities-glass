import pandas as pd 
import numpy as np 

d = pd.read_csv('../data/sim/rct_q12_q13_nn1_s2_t2.csv')
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')

# sanity check 
d1 = d[d['timestep'] == 1]

import configuration as cn 
idx_start = 361984
ConfStart = cn.Configuration(idx_start, configurations, configuration_probabilities)

idx_neighbors, p_neighbors = ConfStart.id_and_prob_of_neighbors()
neighbor_list = []
for idx, p in zip(idx_neighbors, p_neighbors): 
    ConfNeighbor = cn.Configuration(idx, configurations, configuration_probabilities)
    difference = ConfStart.diverge(ConfNeighbor, question_reference)
    question = difference['question'].unique()[0]
    neighbor_list.append((idx, question, p))
neighbor_data = pd.DataFrame(neighbor_list, columns = ['config_id', 'question', 'probability'])

neighbor_data
dt = neighbor_data.merge(d1, on = 'config_id', how = 'inner')
dt
len(dt)
len(d1)
