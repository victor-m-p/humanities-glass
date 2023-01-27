'''
VMP: Updated data. 
Number of fixed traits. 
'''

# COGSCI23
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.lines import Line2D
import arviz as az

# plotting setup
small_text = 12
large_text = 18

# read data
d_enforcement = pd.read_csv('../data/COGSCI23/enforcement_observed.csv')
d_enforcement['prob_remain'] = (1-d_enforcement['prob_move'])*100

# we need the median as well
median_remain = d_enforcement.groupby('n_fixed_traits')['prob_remain'].median().tolist()

# HDI plot 
hdi_list = []
for n_traits in d_enforcement['n_fixed_traits'].unique(): 
    n_traits_subset = d_enforcement[d_enforcement['n_fixed_traits'] == n_traits]
    n_traits_remain = n_traits_subset['prob_remain'].values
    hdi_95 = list(az.hdi(n_traits_remain, hdi_prob = 0.95))     
    hdi_50 = list(az.hdi(n_traits_remain, hdi_prob = 0.5))
    data_list = [n_traits] + hdi_95 + hdi_50 
    hdi_list.append(data_list)
hdi_df = pd.DataFrame(hdi_list, columns = ['n_fixed_traits',
                                           'hdi_95_l',
                                           'hdi_95_u',
                                           'hdi_50_l',
                                           'hdi_50_u'])
n_fixed_traits = hdi_df['n_fixed_traits'].tolist()
hdi_95_l = hdi_df['hdi_95_l'].tolist()
hdi_95_u = hdi_df['hdi_95_u'].tolist()
hdi_50_l = hdi_df['hdi_50_l'].tolist()
hdi_50_u = hdi_df['hdi_50_u'].tolist()

fig, ax = plt.subplots(figsize = (7, 5), dpi = 300)
plt.fill_between(n_fixed_traits, hdi_95_l, hdi_95_u, color = 'tab:blue', alpha = 0.3)
plt.fill_between(n_fixed_traits, hdi_50_l, hdi_50_u, color = 'tab:blue', alpha = 0.6)
plt.plot(n_fixed_traits, median_remain, color = 'tab:red', linewidth = 2)
plt.xticks(np.arange(0, 20, 1))
plt.xlabel('Number of fixed traits', size = small_text)
plt.ylabel('P(remain)', size = small_text)
plt.savefig('../fig/enforcement_hdi.pdf')

# plot lines and median 
fig, ax = plt.subplots(figsize = (7, 5), dpi = 300)
sns.lineplot(
    x = 'n_fixed_traits',
    y = 'prob_remain',
    estimator = None, 
    lw = 1,
    alpha = 0.20,
    data = d_enforcement,
    units = 'config_id'
)
ax.plot(n_fixed_traits, median_remain, color = 'tab:red', linewidth = 2)
plt.xticks(np.arange(0, 20, 1))
plt.xlabel('Number of fixed traits', size = small_text)
plt.ylabel('P(remain)', size = small_text)
plt.savefig('../fig/enforcement_lines.pdf')

# label some points 
d_enforcement_lead = d_enforcement
d_enforcement_lead['prob_remain_next'] = d_enforcement_lead.groupby('config_id')['prob_remain'].shift(-1)
d_enforcement_lead = d_enforcement_lead.drop_duplicates()

# find some candidates 
## go up a lot early 
max_list = []
for n in [1, 3, 6]: 
    d_n = d_enforcement_lead[d_enforcement_lead['n_fixed_traits'] == n]
    d_n['increase'] = d_n['prob_remain_next'] - d_n['prob_remain']
    max_n = d_n[d_n['increase'] == d_n['increase'].max()]
    max_list.append(max_n) 
d_n = d_enforcement_lead.sample(n = 3, random_state = 5)
max_list.append(d_n)
for n in [1, 3, 6]: 
    d_n = d_enforcement_lead[d_enforcement_lead['n_fixed_traits'] == n]
    d_n = d_n[d_n['prob_remain'] == d_n['prob_remain'].max()]
    max_list.append(d_n)
max_df = pd.concat(max_list)
max_df = max_df[['config_id']].drop_duplicates()

## find their name
max_df = d_enforcement_lead.merge(max_df, on = 'config_id', how = 'inner')
entry_maxlikelihood = pd.read_csv('../data/analysis/entry_maxlikelihood.csv')
entry_maxlikelihood = entry_maxlikelihood[['config_id', 'entry_name']]
entry_maxlikelihood = entry_maxlikelihood.groupby('config_id').sample(n=1, random_state = 1)
entry_maxlikelihood = entry_maxlikelihood.merge(max_df, on = 'config_id', how = 'inner')

## now we pick a couple only ... 
upper_line = entry_maxlikelihood[entry_maxlikelihood['config_id'] == 1027975]
lower_line = entry_maxlikelihood[entry_maxlikelihood['config_id'] == 652162]

## plot each of them: 
fig, ax = plt.subplots(figsize = (7, 5), dpi = 300)
plt.fill_between(n_fixed_traits, hdi_95_l, hdi_95_u, color = 'tab:blue', alpha = 0.3)
plt.fill_between(n_fixed_traits, hdi_50_l, hdi_50_u, color = 'tab:blue', alpha = 0.5)
plt.plot(upper_line['n_fixed_traits'].values,
         upper_line['prob_remain'].values, 
         color = '#152238',
         ls = '--')
plt.plot(lower_line['n_fixed_traits'].values, 
         lower_line['prob_remain'].values, 
         color = '#152238',
         ls = '--'
         )
#sns.lineplot(data = entry_maxlikelihood, x = 'n_fixed_traits',
#             y = 'prob_remain', hue = 'entry_name')
plt.plot(n_fixed_traits, median_remain, color = '#152238', linewidth = 2)
plt.xticks(np.arange(0, 20, 1))
plt.xlabel('Number of fixed traits', size = small_text)
plt.ylabel('P(remain)', size = small_text)
ax.legend(bbox_to_anchor=(0.9, -0.2))
plt.savefig('../fig/enforcement_hdi_labels.pdf', bbox_inches = 'tight')

### which religions inhabit the space
lower_id = lower_line['config_id'].tolist()[0]
upper_id = upper_line['config_id'].tolist()[0]
entry_maxlikelihood = pd.read_csv('../data/analysis/entry_maxlikelihood.csv')
entry_maxlikelihood = entry_maxlikelihood[['config_id', 'entry_name']]

lower_line = entry_maxlikelihood[entry_maxlikelihood['config_id'] == lower_id]
upper_line = entry_maxlikelihood[entry_maxlikelihood['config_id'] == upper_id]

### check the Buddhism ### 
x = entry_maxlikelihood[entry_maxlikelihood['n_fixed_traits'] == 19]
x = x[x['prob_remain'] == x['prob_remain'].min()]
buddhism_idx = 978831

import configuration as cn 
from fun import bin_states 
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
n_nodes = 20
configurations = bin_states(n_nodes) 

buddhism = cn.Configuration(buddhism_idx, 
                            configurations,
                            configuration_probabilities)

                            
p_self = buddhism.p
neighbor_id, neighbor_p = buddhism.pid_neighbors(configurations, 
                                configuration_probabilities)

x = sorted(neighbor_p)
y = x[0]
p_self/(p_self+y)
p_self