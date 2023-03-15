import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# questions 
q1 = 11
q2 = 12
n_timesteps = 200
n_simulations = 2000

# read data
sim_data = pd.read_csv('../data/sim/messalians.csv')
sim_data['timestep'] = sim_data['timestep'] - 1
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')

# find the actual configurations & merge 
q1_yes_q2_yes = list(np.where((configurations[:, q1] == 1) & (configurations[:, q2] == 1))[0])
q1_yes_q2_no = list(np.where((configurations[:, q1] == 1) & (configurations[:, q2] == -1))[0])
q1_no_q2_yes = list(np.where((configurations[:, q1] == -1) & (configurations[:, q2] == 1))[0])
q1_no_q2_no = list(np.where((configurations[:, q1] == -1) & (configurations[:, q2] == -1))[0])
d_combinations = pd.DataFrame({
    'config_id': q1_yes_q2_yes + q1_yes_q2_no + q1_no_q2_yes + q1_no_q2_no,
    'combination': ['yy' for _, _ in enumerate(q1_yes_q2_yes)] + ['yn' for _, _ in enumerate(q1_yes_q2_no)] + ['ny' for _, _ in enumerate(q1_no_q2_yes)] + ['nn' for _, _ in enumerate(q1_no_q2_no)],
})
sim_data = sim_data.merge(d_combinations, on = 'config_id', how = 'left').fillna('other')

# create recoded columns
recode_dict1 = {
    'yy': 1,
    'nn': 0,
    'yn': 0,
    'ny': 0
}
recode_dict2 = {
    'yy': 1, 
    'nn': 0,
    'yn': 0.5,
    'ny': 0.5 
}
sim_data['binary_coding'] = [recode_dict1.get(x) for x in sim_data['combination']]
sim_data['nonbinary_coding'] = [recode_dict2.get(x) for x in sim_data['combination']]

#### summary statistics ####
# ...?

#### plots ####

# plot 1 (% of simulations at yy)
# * fraction of simulations at yy for each timestep 
# * an example simulation
# * expected fraction (baseline probability of yy)
p_yy = np.sum(configuration_probabilities[(np.where((configurations[:, q1] == 1) & (configurations[:, q2] == 1))[0])])
sim_data_agg = sim_data.groupby(['starting_config', 'timestep'])['binary_coding'].mean().reset_index(name = 'mean')

fig, ax = plt.subplots()
sns.lineplot(data=sim_data_agg, 
             x='timestep', 
             y='mean',
             color='tab:blue')
sim_data_example = sim_data[sim_data['simulation'] == 2]
sns.lineplot(data=sim_data_example,
            x = 'timestep',
            y = 'binary_coding',
            color='tab:orange')
plt.hlines(p_yy, xmin=0, xmax=200, color='tab:red', linestyle='--')
custom_lines = [Line2D([0], [0], color = 'tab:blue', markersize=10),
                Line2D([0], [0], color = 'tab:orange', markersize=10),
                Line2D([0], [0], color = 'tab:red', markersize=10)]
handles = ['fraction of simulations at yy',
           'example simulation',
           'baseline probability of yy']
plt.suptitle('Messalian: fraction of simulations at yy')
plt.xlabel('n(timestep)')
plt.ylabel('fraction of simulations at yy')
ax.legend(custom_lines, handles, fontsize = 8)
plt.tight_layout()
plt.savefig('../fig/intervention/messalian_fraction.png')

# plot 2 (time to first acquisition)
## what is the most common path to first acquisition?
shift_data = sim_data
shift_data = shift_data.rename(columns = {'config_id': 'config_to'})
shift_data = shift_data.convert_dtypes()
shift_data['config_from'] = shift_data.groupby(['simulation', 'starting_config'])['config_to'].shift(1)
shift_data = shift_data.dropna()

## plot 2.1 (time to first acquisition; including steps not taken)
### Find the first occurrence of 1 for each group
timesteps_first = shift_data[shift_data['binary_coding'] == 1]
timesteps_first = timesteps_first.groupby('simulation')['timestep'].min().reset_index(name = 'steps')
timesteps_first = timesteps_first.groupby('steps').size().reset_index(name = 'count')
timesteps_first = timesteps_first.sort_values('steps', ascending = True)

### create grid 
max_n_steps = timesteps_first['steps'].max()
max_grid = pd.DataFrame({'steps': np.arange(0, max_n_steps + 1)})
timesteps_first = timesteps_first.convert_dtypes()
timesteps_first = max_grid.merge(timesteps_first, on = 'steps', how = 'left').fillna(0)
timesteps_first['percent'] = (timesteps_first['count'] / n_simulations)*100

### plot 
fig, ax = plt.subplots()
sns.lineplot(data=timesteps_first, x='steps', y='percent')
plt.suptitle('Number of (time) steps to first acquisition')
plt.xlabel('n(steps)')
plt.ylabel('%(simulations)')
plt.tight_layout()
plt.savefig('../fig/intervention/messalians_timesteps_first.png')

## plot 2.2 (time to first acquisition; only steps taken)
takensteps_first = shift_data.drop_duplicates(subset=['simulation', 'starting_config', 'config_to'], keep='first')
takensteps_first = takensteps_first[takensteps_first['config_from'] != takensteps_first['config_to']] # just for consistency right now
takensteps_first['count'] = takensteps_first.groupby('simulation')['binary_coding'].cumsum()
takensteps_first = takensteps_first[takensteps_first['count'] <= 1].drop('count', axis=1)
takensteps_first = takensteps_first.groupby('simulation').filter(lambda x: (x['binary_coding'] != 0).any())

### length of paths ### 
# definitely undersampled here.
# we need more simulations for sure
# e.g. n = 10K or something crazy higher than current. 
# could start with n = 1K but still might be sparse...
takensteps_first = takensteps_first.groupby('simulation').size().reset_index(name = 'steps')
takensteps_first = takensteps_first.groupby('steps').size().reset_index(name = 'count')
takensteps_first = takensteps_first.sort_values('steps', ascending = True)

# create grid 
max_n_steps = takensteps_first['steps'].max()
max_grid = pd.DataFrame({'steps': np.arange(0, max_n_steps + 1)})
takensteps_first = takensteps_first.convert_dtypes()
takensteps_first = max_grid.merge(takensteps_first, on = 'steps', how = 'left').fillna(0)
takensteps_first['percent'] = (takensteps_first['count'] / n_simulations)*100

fig, ax = plt.subplots()
sns.lineplot(data=takensteps_first, x='steps', y='percent')
plt.suptitle('Number of (taken) steps to first acquisition')
plt.xlabel('n(steps)')
plt.ylabel('n(simulations)')
plt.tight_layout()
plt.savefig('../fig/intervention/messalians_takensteps_first.png') 

## plot 3 (first transition most often leading to target) 
### issue here ... 
first_transition = shift_data.groupby('simulation').first().reset_index()
first_transition = first_transition.groupby('config_to').size().reset_index(name = 'count')
first_transition = first_transition.sort_values('count', ascending = False)
first_transition = first_transition.rename(columns = {'config_to': 'config_id'})
first_transition = first_transition.merge(d_combinations, on = 'config_id', how = 'left')

### need to merge with neighbors to get question
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

first_transition = neighbor_data.merge(first_transition, on = 'config_id', how = 'left').fillna(0)
first_transition = first_transition.sort_values('count', ascending = False)
first_transition['percent'] = (first_transition['count'] / n_simulations)*100

### plot 
fig, ax = plt.subplots()
sns.barplot(data=first_transition, x='percent', y='question')
plt.suptitle('Most common first transition towards target')
plt.xlabel('n(simulations)')
plt.ylabel('')
plt.tight_layout()
plt.savefig('../fig/intervention/messalians_first_flip.png')
