import pandas as pd 
import numpy as np 

# loads 
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype=int)
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')

# setup 
intervention_var = 1 # scriptures
outcome_var = 12 # punishment

# sample 1K configurations with both == -1
total_pop = np.where((configurations[:, intervention_var] == -1) & (configurations[:, outcome_var] == -1))[0]
probability_pop = configuration_probabilities[total_pop]
probability_pop_norm = probability_pop / probability_pop.sum()
control_pop_idx = np.random.choice(total_pop, 1000, replace=False, p=probability_pop_norm)
control_pop_config = configurations[control_pop_idx]

# now intervene on half of these
experiment_pop_config = np.copy(control_pop_config)
experiment_pop_config[:, intervention_var] = experiment_pop_config[:, intervention_var]*-1
experiment_pop_idx = np.array([np.where((configurations == i).all(1))[0][0] for i in experiment_pop_config])

# save the samples
np.savetxt(f"../data/RCT/experiment.pop.{intervention_var+1}.{outcome_var+1}.txt", experiment_pop_idx, fmt='%i')
np.savetxt(f"../data/RCT/control.pop.{intervention_var+1}.{outcome_var+1}.txt", control_pop_idx, fmt='%i')