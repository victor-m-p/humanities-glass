import itertools 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# read data 
d = pd.read_csv('../data/reference/direct_reference_questions_20_maxna_5_nrows_455_entries_407.csv')
# only work with complete data 
d = d[d['weight'] > 0.99]
d_questions = d.drop(columns=['entry_id', 'weight'])
A = np.array(d_questions)
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(A, ax=ax, cmap = 'RdYlGn', cbar=False)
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.xlabel('Questions', size = 20)
plt.ylabel('Religions', size = 20)
plt.savefig('../fig/heatmap_questions.pdf')
plt.savefig('../fig/heatmap_questions.svg')

# find the most common just in our raw data
d_columns = d_questions.columns.tolist()
d_most_common = d_questions.groupby(d_columns).size().reset_index(name = 'count').sort_values('count', ascending=False)
d_five_most_common = d_most_common.drop(columns=['count']).head(5)
A = np.array(d_five_most_common)
fig, ax = plt.subplots(figsize=(10, 2.5))
sns.heatmap(A, linewidths = 1, ax=ax, cbar=False, cmap = 'RdYlGn')
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.xlabel('Questions', size = 20)
plt.ylabel('Religions', size = 20)
plt.savefig('../fig/heatmap_observed.pdf')
plt.savefig('../fig/heatmap_observed.svg')

# Naive method
# define a mapping for the recoding
mapping = {-1: 0, 0: 0.5, 1: 1}
d_questions = d_questions.replace(mapping)
B = np.array(d_questions)
n_nodes = 20
h_combinations = np.array(list(itertools.product([1, -1], repeat = n_nodes)))
p_naive = np.mean(B, axis=0)

# Create boolean mask
mask = (h_combinations == 1)

# Create array of 1 - B
B_inv = 1 - p_naive

# Perform element-wise multiplication
C = p_naive * mask + B_inv * (~mask)
row_p = np.prod(C, axis=1)
np.sum(row_p) # fuck yes. 

high_p = row_p.argsort()[-5:][::-1]
naive_comb = h_combinations[high_p]
fig, ax = plt.subplots(figsize=(10, 2.5))
sns.heatmap(naive_comb, linewidths = 1, ax=ax, cbar=False, cmap = 'RdYlGn')
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.xlabel('Questions', size = 20)
plt.ylabel('Religions', size = 20)
plt.savefig('../fig/heatmap_naive.pdf')
plt.savefig('../fig/heatmap_naive.svg')

#p_true[p_true >= 0.5] = 1
#p_true[p_true < 0.5] = -1 # this is the string according to just means 

# what is actually the most "probable configuration"? 
# what is the least probable configuration? 
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype=int)
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
max_index = configuration_probabilities.argsort()[-5:][::-1]
most_common = configurations[max_index]

### tmp
min_index = configuration_probabilities.argmin()
configurations[min_index]

fig, ax = plt.subplots(figsize=(10, 2.5))
sns.heatmap(most_common, linewidths = 1, ax=ax, cbar=False, cmap = 'RdYlGn')
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.xlabel('Questions', size = 20)
plt.ylabel('Religions', size = 20)
plt.savefig('../fig/heatmap_model.pdf')
plt.savefig('../fig/heatmap_model.svg')

# 1, 1, 0, 0
# 1, 0, 0
# 1, 1, -1
# -1, -1, 1
# 1, -1, 1
# -1, -1, 1, -1

# distribution of p 
naive_sorted = np.sort(row_p)
model_sorted = np.sort(configuration_probabilities)

#all_probabilities = np.stack(all_probabilities)
n_states = len(naive_sorted)
plt.plot([i for i in range(n_states)],
         [np.log(i*(n_states)) for i in naive_sorted],
         color = 'tab:orange', lw = 3)
plt.plot([i for i in range(n_states)],
         [np.log(i*(n_states)) for i in model_sorted],
         color = 'tab:blue', lw = 3)

