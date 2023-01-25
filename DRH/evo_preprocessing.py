# COGSCI23
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import os 

# read all files 
path = '../data/COGSCI23/evo_raw'
dir_list = os.listdir(path)

### subsample every 10 steps ###
list_from = []
list_to = []
for filename in dir_list:
    d = pd.read_csv(f'{path}/{filename}')
    for t2 in range(11, 101, 10): 
        d_from = d[d['timestep'] == 1]
        d_to = d[d['timestep'] == t2]
        d_from = d_from.rename(columns = {'config_id': 'config_from',
                                          'timestep': 't_from'})
        d_from = d_from[['config_from', 't_from']]
        d_to = d_to.rename(columns = {'config_id': 'config_to',
                                      'timestep': 't_to'})
        d_to = d_to[['config_to', 't_to']]
        list_from.append(d_from)
        list_to.append(d_to)

# wrangle
d_from = pd.concat(list_from)
d_to = pd.concat(list_to)
d_edgelist = pd.concat([d_from.reset_index(drop=True),
                        d_to.reset_index(drop=True)],
                        axis = 1)

# save 
d_edgelist_save = d_edgelist.drop(columns = ['t_from'])
d_edgelist_save.to_csv('../data/COGSCI23/evo_clean/subsample.csv', index = False)

### collapse to mean ###
n_simulations = 100
n_timesteps = 101
list_agg = []
for filename in dir_list:
    d = pd.read_csv(f'{path}/{filename}')
    # add config_from  
    d_from = d[d['timestep'] == 1]
    start_config = d_from['config_id'].unique().tolist()
    len_start_config = len(start_config)
    start_config = [[config for _ in range(n_simulations*n_timesteps)] for config in start_config]
    start_config = [item for sublist in start_config for item in sublist]
    start_config = pd.DataFrame({'config_from': start_config})
    # concat with data 
    d = pd.concat([d, start_config], axis = 1)
    # exclude timestep == 1 because that is given 
    d = d[d['timestep'] > 1]
    d = d.rename(columns = {'config_id': 'config_to'})
    d_weight = d.groupby(['config_from', 'config_to']).size().reset_index(name = 'weight')
    # add to lists 
    list_agg.append(d_weight)
d_weight = pd.concat(list_agg)
d_weight.to_csv('../data/COGSCI23/evo_clean/weighted.csv', index = False)

### shift data ###
n_simulations = 100
n_timesteps = 101
for filename in dir_list: 
    d = pd.read_csv(f'{path}/{filename}')
    # add config_from
    d_from = d[d['timestep'] == 1]
    start_config = d_from['config_id'].unique().tolist()
    len_start_config = len(start_config)
    start_config = [[config for _ in range(n_simulations*n_timesteps)] for config in start_config]
    start_config = [item for sublist in start_config for item in sublist]
    start_config = pd.DataFrame({'config_orig': start_config})
    # concat with data 
    d = pd.concat([d, start_config], axis = 1)
    # shift data
    d = d.rename(columns = {'config_id': 'config_from'})
    d = d.sort_values(['config_orig', 'simulation', 'timestep'],
                      ascending = [True, True, True])
    d = d.convert_dtypes()
    d['config_to'] = d.groupby(['simulation', 'config_orig'])['config_from'].shift(-1)
    d = d.dropna()
    # add to list 
    d.to_csv(f'../data/COGSCI23/evo_shift/shift_{filename}', index = False)

### collapse shifted data ###
# read all files 
path = '../data/COGSCI23/evo_shift'
dir_list = os.listdir(path)
list_agg = []
for filename in dir_list: 
    d_shift = pd.read_csv(f"{path}/{filename}")
    d_shift = d_shift.groupby(['config_from', 'config_to']).size().reset_index(name = 'weight')
    list_agg.append(d_shift)
d_shift_weight = pd.concat(list_agg)
d_shift_weight.to_csv(f'../data/COGSCI23/evo_clean/shift_weighted.csv', index = False)

### hamming distance ###
weighted = pd.read_csv('../data/COGSCI23/evo_clean/weighted.csv')

# compute hamming
d_pairs = weighted[['config_from', 'config_to']].drop_duplicates()

# hamming 
import configuration as cn 
from fun import bin_states 

# preparation
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
n_nodes = 20
configurations = bin_states(n_nodes) 

# get all of the hamming distances 
# NB: this is crazy fast, what the fuck?
hamming_list = []
for num, row in tqdm(d_pairs.iterrows()):
    config_from = row['config_from']
    config_to = row['config_to']
    ConfFrom = cn.Configuration(config_from, configurations, 
                               configuration_probabilities)
    ConfTo = cn.Configuration(config_to, configurations, 
                              configuration_probabilities)
    h_distance = ConfFrom.hamming_distance(ConfTo)
    hamming_list.append((config_from, config_to, h_distance))

# bind back onto the original data 
d_hamming = pd.DataFrame(hamming_list, 
                         columns = ['config_from', 'config_to', 'hamming_dist'])

# save 
d_hamming.to_csv('../data/COGSCI23/evo_clean/hamming.csv', index = False)
