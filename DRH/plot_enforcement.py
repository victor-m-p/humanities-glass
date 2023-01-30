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

# label some points 
highlight_configs = [1027975, 652162]
entry_maxlikelihood = pd.read_csv('../data/analysis/entry_maxlikelihood.csv')
entry_maxlikelihood = entry_maxlikelihood[['config_id', 'entry_name']]
entry_maxlikelihood = entry_maxlikelihood[entry_maxlikelihood['config_id'].isin(highlight_configs)]
entry_sample = entry_maxlikelihood.groupby('config_id').sample(n=1, random_state = 1)
entry_sample = entry_sample.merge(d_enforcement, on = 'config_id', how = 'inner')
upper_line = entry_sample[entry_sample['config_id'] == highlight_configs[0]]
lower_line = entry_sample[entry_sample['config_id'] == highlight_configs[1]]

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

# hdi plot
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
plt.plot(n_fixed_traits, median_remain, color = '#152238', linewidth = 2)
plt.xticks(np.arange(0, 20, 1))
plt.xlabel('Number of fixed traits', size = small_text)
plt.ylabel('P(remain)', size = small_text)
ax.legend(bbox_to_anchor=(0.9, -0.2))
plt.savefig('../fig/enforcement/enforcement_hdi.pdf', bbox_inches = 'tight')

# lines plot 
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
plt.plot(upper_line['n_fixed_traits'].values,
         upper_line['prob_remain'].values, 
         color = '#152238',
         ls = '--')
plt.plot(lower_line['n_fixed_traits'].values, 
         lower_line['prob_remain'].values, 
         color = '#152238',
         ls = '--'
         )
plt.plot(n_fixed_traits, median_remain, color = '#152238', linewidth = 2)
plt.xticks(np.arange(0, 20, 1))
plt.xlabel('Number of fixed traits', size = small_text)
plt.ylabel('P(remain)', size = small_text)
plt.savefig('../fig/enforcement/enforcement_lines.pdf')