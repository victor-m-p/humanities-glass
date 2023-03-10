import pandas as pd 
import numpy as np 
from tqdm import tqdm

configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype=int)
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')

# function to calculate pairs 
def calculate_pairs(configurations, configuration_probabilities, q1, q2): 
    # get probability of combinations
    q1_yes_q2_yes = configuration_probabilities[np.where((configurations[:, q1] == 1) & (configurations[:, q2] == 1))[0]].sum()
    q1_yes_q2_no = configuration_probabilities[np.where((configurations[:, q1] == 1) & (configurations[:, q2] == -1))[0]].sum()
    q1_no_q2_yes = configuration_probabilities[np.where((configurations[:, q1] == -1) & (configurations[:, q2] == 1))[0]].sum()
    q1_no_q2_no = configuration_probabilities[np.where((configurations[:, q1] == -1) & (configurations[:, q2] == -1))[0]].sum()
    return q1_yes_q2_yes, q1_yes_q2_no, q1_no_q2_yes, q1_no_q2_no

yy_o, yn_o, ny_o, nn_o = calculate_pairs(configurations, p_old, 11, 12)
yy_n, yn_n, ny_n, nn_n = calculate_pairs(configurations, configuration_probabilities, 11, 12)

# this produces Simon results
idx_init = 11
restricted_configurations = np.delete(configurations, [idx_init, idx_init + 1], 1)
restricted_configurations = np.unique(restricted_configurations, axis = 0)

n_configurations = len(restricted_configurations)
samples = 1000
sample_config_idx = np.random.choice(n_configurations,
                                     size = samples,
                                     replace = False)
sample_configs = restricted_configurations[sample_config_idx]

transition_probabilities = []
for num, x in tqdm(enumerate(sample_configs)): 
    # get the configurations 
    conf_both = np.insert(x, idx_init, [1, 1])
    conf_none = np.insert(x, idx_init, [-1, -1])
    conf_first = np.insert(x, idx_init, [1, -1])
    conf_second = np.insert(x, idx_init, [-1, 1])
    # get the configuration idx 
    p_both = configuration_probabilities[np.where(np.all(configurations == conf_both, axis = 1))[0][0]]
    p_none = configuration_probabilities[np.where(np.all(configurations == conf_none, axis = 1))[0][0]]
    p_first = configuration_probabilities[np.where(np.all(configurations == conf_first, axis = 1))[0][0]]
    p_second = configuration_probabilities[np.where(np.all(configurations == conf_second, axis = 1))[0][0]]
    # get probabilities
    transition_probabilities.append((p_both, p_none, p_first, p_second))

d = pd.DataFrame(transition_probabilities, columns = ['yy', 'nn', 'yn', 'ny'])

row_sums = d.sum(axis=1)
df_normalized = d.divide(row_sums, axis=0)
df_normalized.mean()