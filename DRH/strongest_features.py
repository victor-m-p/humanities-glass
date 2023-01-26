import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

# preprocessing 
from fun import bin_states 
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
n_nodes = 20
configurations = bin_states(n_nodes) 

# find probability for different attributes
probability_list = []
for i in range(20): 
    column_n = configurations[:, i]
    column_n_idx = np.where(column_n > 0)
    column_probs = configuration_probabilities[column_n_idx]
    mean_prob = np.mean(column_probs)
    std_prob = np.std(column_probs)
    probability_list.append((i+1, mean_prob, std_prob))
probability_df = pd.DataFrame(probability_list, columns = ['question_id', 'mean(prob)', 'std(prob)'])

# match with questions 
question_reference = pd.read_csv('../data/analysis/question_reference.csv')
question_reference = question_reference[['question_id', 'question']]
question_probability = question_reference.merge(probability_df, on = 'question_id', how = 'inner')
question_probability = question_probability.sort_values('mean(prob)').reset_index()

fig, ax = plt.subplots(dpi = 300)
for i, row in question_probability.iterrows(): 
    x = row['mean(prob)']
    x_err = row['std(prob)']
    plt.scatter(x, i, color = 'tab:blue')
plt.yticks(np.arange(0, 20, 1), question_probability['question'].values)
plt.xlabel('Mean probability')
plt.savefig('../fig/feature_stability.pdf')