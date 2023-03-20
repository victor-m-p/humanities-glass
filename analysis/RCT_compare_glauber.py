import pandas as pd 
import numpy as np
import seaborn as sns 
import os 
import matplotlib.pyplot as plt 

# setup
n_timesteps = 100

# configurations 
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype=int)

# get the n=1 files 
basepath = '../data/RCT/'
files = os.listdir(basepath)
files = [f for f in files if f.endswith('.csv') and '.5.10.' in f]

data_list = []
for f in files: 
    d = pd.read_csv(basepath + f)
    d['condition'] = f.split('.')[0]
    d['n_enforced'] = f.split('.')[1]  
    d['n_free'] = f.split('.')[2]
    d['intervention_var'] = f.split('.')[3]
    d['outcome_var'] = f.split('.')[4]
    data_list.append(d)
d = pd.concat(data_list)    
d['identifier'] = d['condition'] + '_' + d['n_enforced'] + '_' + d['n_free']
d['trial'] = d['intervention_var'] + '_' + d['outcome_var']
d = d[d['n_free'] != '95'] # remove the 95% free condition

# get the 50% files 
basepath = '../data/RCT2/'
files = os.listdir(basepath)
files = [f for f in files if f.endswith('.csv') and '.5.10.' in f]

data_list = []
for f in files: 
    d2 = pd.read_csv(basepath + f)
    d2['condition'] = f.split('.')[1]
    d2['n_enforced'] = f.split('.')[2]  
    d2['n_free'] = f.split('.')[3]
    d2['intervention_var'] = f.split('.')[4]
    d2['outcome_var'] = f.split('.')[5]
    data_list.append(d2)
d2 = pd.concat(data_list)    
d2['identifier'] = d2['condition'] + '_' + d2['n_enforced'] + '_' + d2['n_free']
d2['trial'] = d2['intervention_var'] + '_' + d2['outcome_var']

# groupby timestep and look at difference
d_grouped = d.groupby(['timestep', 'identifier'])['outcome'].mean().reset_index()
d_baseline = d_grouped[d_grouped['identifier'] == 'control_0_100']
d_baseline = d_baseline.rename(columns={'outcome': 'outcome_baseline',
                                        'identifier': 'identifier_baseline'})
d_agg = d_grouped.merge(d_baseline, on='timestep', how ='inner')
d_agg['difference'] = d_agg['outcome'] / d_agg['outcome_baseline']

d2_grouped = d2.groupby(['timestep', 'identifier'])['outcome'].mean().reset_index()
d2_baseline = d2_grouped[d2_grouped['identifier'] == 'control_0_100']
d2_baseline = d2_baseline.rename(columns={'outcome': 'outcome_baseline',
                                          'identifier': 'identifier_baseline'})
d2_agg = d2_grouped.merge(d2_baseline, on='timestep', how ='inner')
d2_agg['difference'] = d2_agg['outcome'] / d2_agg['outcome_baseline']                                

d2_agg = d2_agg.sort_values('identifier')
d_agg = d_agg.sort_values('identifier')
d = d.sort_values('identifier')
d2 = d2.sort_values('identifier')

#### plot DIVIDE BY BASELINE #### 
fig, ax = plt.subplots(2, sharey=True)

sns.lineplot(data=d_agg,
             x='timestep',
             y='difference',
             hue='identifier',
             ax=ax[0])
sns.lineplot(data=d2_agg,
             x='timestep',
             y='difference',
             hue='identifier',
             ax=ax[1])

# Set the same ylim for both subplots
plt.ylim(0.6, 2)

# Set the common ylabel for both subplots, adjust position to avoid overlapping with axis text
ax[0].set_ylabel('P(Y=1|condition) / P(Y=1|baseline)', labelpad=15)
ax[0].yaxis.set_label_coords(-.1, -.1)
ax[1].set_ylabel('')

# Add a common legend below the subplots
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1), borderaxespad=0.)

# Remove the legend from each subplot
ax[0].legend_.remove()
ax[1].legend_.remove()

# Remove the xlabel for the first subplot
ax[0].set_xlabel('')

# Set the xlabel for the second subplot
ax[1].set_xlabel('timestep')

plt.suptitle('Formal Burials ON Grave Goods')
plt.show();


#### PLOT THE RAW #### 
fig, ax = plt.subplots(2, sharey=True)

sns.lineplot(data=d,
             x='timestep',
             y='outcome',
             hue='identifier',
             ax=ax[0])
sns.lineplot(data=d2,
             x='timestep',
             y='outcome',
             hue='identifier',
             ax=ax[1])

# Set the same ylim for both subplots
plt.ylim(0, 0.8)

# Set the common ylabel for both subplots, adjust position to avoid overlapping with axis text
ax[0].set_ylabel('%Y=1', labelpad=15)
ax[0].yaxis.set_label_coords(-.1, -.1)
ax[1].set_ylabel('')

# Add a common legend below the subplots
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1), borderaxespad=0.)

# Remove the legend from each subplot
ax[0].legend_.remove()
ax[1].legend_.remove()

# Remove the xlabel for the first subplot
ax[0].set_xlabel('')

# Set the xlabel for the second subplot
ax[1].set_xlabel('timestep')

plt.suptitle('Formal Burials ON Grave Goods')
plt.show();

## need to check whether I made mistake 
## could also just be much harder to understand what is happening 
## and we need to run radically more iterations to get a robust sense 
