import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

d_evo = pd.read_csv('../data/COGSCI23/evolution_maxlik_s_100_t_10.csv')

# 1871/4683 unique 
len(d_evo['config_id'].unique())

# (1) where do they end up 
d_sub = d_evo[(d_evo['timestep'] == 1) | (d_evo['timestep'] == 10)]

## (NB: find better way)
d_from = d_sub[d_sub['timestep'] == 1]
d_to = d_sub[d_sub['timestep'] == 10]
d_from = d_from.rename(columns = {'config_id': 'config_from'})
d_to = d_to.rename(columns = {'config_id': 'config_to'})

d_edgelist = pd.concat([d_from.reset_index(drop=True), 
                        d_to.reset_index(drop=True)], 
                       axis=1)

## if this looks good, we can get rid of everything else
d_edgelist = d_edgelist[['config_from', 'config_to']]

## visualize as a directed network 
w_edgelist = d_edgelist.groupby(['config_from', 'config_to']).size().reset_index(name = 'weight')
w_edgelist.sort_values('weight', ascending = False)


### number for MAX neighbor
fig, ax = plt.subplots()
wc_edgelist = w_edgelist.groupby('config_from')['weight'].max().reset_index(name = 'maximum')
max_val = wc_edgelist['maximum'].max()
min_val = wc_edgelist['maximum'].min()
n_bins = max_val - min_val
val_width = max_val - min_val
bin_width = val_width/n_bins 
sns.histplot(data = wc_edgelist, x = 'maximum', 
             bins = n_bins+1,
             binrange=(min_val, max_val+1))
plt.xticks(ticks = np.arange(min_val-bin_width/2, max_val+bin_width/2, bin_width),
           labels = np.arange(min_val, max_val+bin_width, bin_width))

labels = np.arange(min_val, max_val+bin_width, bin_width)
ticks = np.arange(min_val-bin_width/2, max_val+bin_width/2, bin_width)
ticks
len(ticks)
len(labels)

wc_edgelist.groupby('maximum').size()
n_bins
np.arange(min_val, max_val+bin_width, bin_width)
min_val
max_val
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("right")

plt.xticks(np.arange(min_val-bin_width/2, max_val+bin_width/2, bin_width))


### distribution of connections  
sns.histplot(data = w_edgelist, x = 'weight')
plt.xticks(np.arange(min_val-bin_width, max_val+bin_width, bin_width))
plt.xlabel('n any neighbor')

## 500... it is quite a lot 
## we have way too many ties 
## to just take the maximum ...
## let it run a bit to see. 