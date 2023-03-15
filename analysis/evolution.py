import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels

# questions 
q1 = 11
q2 = 12

# read data
sim_data = pd.read_csv('../data/sim/messalians.csv')
sim_data['timestep'] = sim_data['timestep'] - 1
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')

# find the configurations that are actually "successes";
network_information = pd.read_csv('../data/analysis/top_configurations_network.csv')
network_information = network_information[['config_id', 'config_prob', 'entry_name']]
network_config_id = network_information['config_id'].tolist()
network_configs = configurations[network_config_id]

q1_label = question_reference[question_reference['question_id'] == q1+1]['question_short'].values[0]
q2_label = question_reference[question_reference['question_id'] == q2+1]['question_short'].values[0]

# find the actual configurations
#top_configs = configurations[network_information['config_id'].values, :]
q1_yes_q2_yes = list(np.where((configurations[:, q1] == 1) & (configurations[:, q2] == 1))[0])
q1_yes_q2_no = list(np.where((configurations[:, q1] == 1) & (configurations[:, q2] == -1))[0])
q1_no_q2_yes = list(np.where((configurations[:, q1] == -1) & (configurations[:, q2] == 1))[0])
q1_no_q2_no = list(np.where((configurations[:, q1] == -1) & (configurations[:, q2] == -1))[0])

d_combinations = pd.DataFrame({
    'config_id': q1_yes_q2_yes + q1_yes_q2_no + q1_no_q2_yes + q1_no_q2_no,
    'combination': ['yy' for _, _ in enumerate(q1_yes_q2_yes)] + ['yn' for _, _ in enumerate(q1_yes_q2_no)] + ['ny' for _, _ in enumerate(q1_no_q2_yes)] + ['nn' for _, _ in enumerate(q1_no_q2_no)],
})

#d_combinations = d_combinations.sort_values('index').reset_index(drop = True)
#network_information = pd.merge(network_information, d_combinations, left_index=True, right_index=True)

# merge with the occurences
# NB: hard to show the "getting closer" thing
# i.e. getting one of the properties ...  
#network_positive = network_information[['config_id', 'combination']]
#network_positive = network_positive[network_positive['combination'] == 'yy']
sim_data = sim_data.merge(d_combinations, on = 'config_id', how = 'left').fillna('other')

# plot some transitions 
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

# what is the yy probability mass?
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
p_yy = np.sum(configuration_probabilities[(np.where((configurations[:, q1] == 1) & (configurations[:, q2] == 1))[0])])

# plot transitions 
# messalian; in general when they get there, they'll stay there
# but it is also true that being at yy has high baseline probability 
sim_data_agg = sim_data.groupby(['starting_config', 'timestep'])['binary_coding'].mean().reset_index(name = 'mean')
sim_data_agg['starting_config'] = sim_data_agg['starting_config'].astype(str)
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
# would be interesting to plot distance
# but I guess that unless we are aiming
# for a particular configuration the best
# we can do is the (flip one) thing. 

# what is the most common path to first acquisition?
shift_data = sim_data
shift_data = shift_data.rename(columns = {'config_id': 'config_to'})
shift_data = shift_data.convert_dtypes()
shift_data['config_from'] = shift_data.groupby(['simulation', 'starting_config'])['config_to'].shift(1)
shift_data = shift_data.dropna()

### first kind of plot 
# Find the first occurrence of 1 for each group
first_occurrence = shift_data[shift_data['binary_coding'] == 1]
first_occurrence = first_occurrence.groupby('simulation')['timestep'].min().reset_index(name = 'steps')
first_occurrence = first_occurrence.groupby('steps').size().reset_index(name = 'count')
first_occurrence = first_occurrence.sort_values('steps', ascending = True)

# create grid 
max_n_steps = first_occurrence['steps'].max()
max_grid = pd.DataFrame({'steps': np.arange(0, max_n_steps + 1)})
first_occurrence = first_occurrence.convert_dtypes()
first_occurrence = max_grid.merge(first_occurrence, on = 'steps', how = 'left').fillna(0)

fig, ax = plt.subplots()
sns.lineplot(data=first_occurrence, x='steps', y='count')
#sns.regplot(data=first_occurrence, x='steps', y='count', 
#            scatter=False, color='tab:orange',
#            lowess=True)
plt.suptitle('Number of (time) steps to first acquisition')
plt.xlabel('n(steps)')
plt.ylabel('n(simulations)')
plt.tight_layout()
plt.savefig('../fig/intervention/messalians_first_timestep.png')


### second kind of plot 
shift_test = shift_data.drop_duplicates(subset=['simulation', 'starting_config', 'config_to'], keep='first')
shift_test = shift_test[shift_test['config_from'] != shift_test['config_to']] # just for consistency right now
# remove all columns after encountering first 1 
shift_test['count'] = shift_test.groupby('simulation')['binary_coding'].cumsum()
shift_test = shift_test[shift_test['count'] <= 1].drop('count', axis=1)
# remove all simulations that do not hit target
shift_test = shift_test.groupby('simulation').filter(lambda x: (x['binary_coding'] != 0).any())

### unique paths ###
# find out what to do with this
unique_paths = shift_test.groupby('simulation')['config_to'].agg(lambda x: ' '.join(str(i) for i in x)).reset_index()
unique_paths = unique_paths.groupby('config_to').size().reset_index(name = 'count')
unique_paths = unique_paths.sort_values('count', ascending=False) # still very sparse 
unique_paths.head(5)

### length of paths ### 
# definitely undersampled here.
# we need more simulations for sure
# e.g. n = 10K or something crazy higher than current. 
# could start with n = 1K but still might be sparse...
n_steps_to_first = shift_test.groupby('simulation').size().reset_index(name = 'steps')
n_steps_to_first = n_steps_to_first.groupby('steps').size().reset_index(name = 'count')
n_steps_to_first = n_steps_to_first.sort_values('steps', ascending = True)

# create grid 
max_n_steps = n_steps_to_first['steps'].max()
max_grid = pd.DataFrame({'steps': np.arange(0, max_n_steps + 1)})
n_steps_to_first = n_steps_to_first.convert_dtypes()
n_steps_to_first = max_grid.merge(n_steps_to_first, on = 'steps', how = 'left').fillna(0)

fig, ax = plt.subplots()
sns.lineplot(data=n_steps_to_first, x='steps', y='count')
plt.suptitle('Number of (taken) steps to first acquisition')
plt.xlabel('n(steps)')
plt.ylabel('n(simulations)')
plt.tight_layout()
plt.savefig('../fig/intervention/messalians_first_takenstep.png') 

### most common first acquisition towards target ### 
# both "ny" are present, but one of them very improbable
# because it is just not a very likely path
# so the bit is hard to flip probably 
first_transition = shift_test.groupby('simulation').first().reset_index()
first_transition = first_transition.groupby('config_to').size().reset_index(name = 'count')
first_transition = first_transition.sort_values('count', ascending = False)
first_transition = first_transition.rename(columns = {'config_to': 'config_id'})
first_transition = first_transition.merge(d_combinations, on = 'config_id', how = 'left')

## we should generate the last two
# i.e. what are all the neighbors?

# plot this; what do they correspond to?
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
fig, ax = plt.subplots()
sns.barplot(data=first_transition, x='count', y='question')
plt.suptitle('Most common first transition towards target')
plt.xlabel('n(simulations)')
plt.ylabel('')
plt.tight_layout()
plt.savefig('../fig/intervention/messalians_traits.png')

### stability ###
# a bit more involved ...

### returning? ###

### next things to do ### 
# run for higher n (tonight) and slightly longer 
# test for which paths lead to stability (not just attainment)
# move on to the other part of the puzzle (RCT). 
# (need more direct interpretation) and potentially visualization (e.g. network). 