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

############# old stuff ###########

# curves for each community 
## either the two small ones are just different
## or this might suggest that that curve is the MEDIAN
## while for the other ones it is driven up a bit because
## of outliers?
network_information = pd.read_csv('../data/analysis/network_information_enriched.csv')
community_information = network_information[['config_id', 'comm_label']]
community_enforcement = d_enforcement.merge(community_information, on = 'config_id', how = 'inner')
community_enforcement = community_enforcement.sort_values('comm_label', ascending = True)

fig, ax = plt.subplots(figsize = (7, 5), dpi = 300)
sns.pointplot(data = community_enforcement, x = 'n_fixed_traits', 
              y = 'prob_remain', hue = 'comm_label')
plt.suptitle('Stability (by community)', size = large_text)
plt.xlabel('Number of fixed traits', size = small_text)
plt.ylabel('Probability remain', size = small_text)
plt.legend(title = 'Community')
plt.savefig('../fig/COGSCI23/stability/community.pdf')

# plot a couple of interesting ones (what is interesting?) 
## (1) one that is just very stable 
## (2) one that is very unstable, but where enforcement is effective
## (3) one that is very unstable but where enforcement is effective 
## ... find out which religions they correspond to ... 
## (4) another way to find "interesting" of course is to look 
## for theoretically interesting religions. 
## NB: lowest std. removed because that equals highest sum. 

# find based on overall sum
config_group_sum = d_enforcement.groupby('config_id')['prob_remain'].sum().reset_index(name = 'sum')
## lowest sum
config_min_sum = config_group_sum.sort_values('sum', ascending = True).head(1)
## highest sum
config_max_sum = config_group_sum.sort_values('sum', ascending = False).head(1)
# find based on overall std 
config_group_std = d_enforcement.groupby('config_id')['prob_remain'].std().reset_index(name = 'std')
## highest std
config_max_std = config_group_std.sort_values('std', ascending = False).head(1)

# preparation
configs = [config_min_sum,
           config_max_sum, 
           config_max_std]

colors = ['tab:blue',
          'tab:orange',
          'tab:red']

# plot these four 
fig, ax = plt.subplots(figsize = (7, 5), dpi = 300)
for config, color in zip(configs, colors):
    tmp_config = d_enforcement.merge(config, on = 'config_id', how = 'inner')
    x = tmp_config['n_fixed_traits'].tolist()
    y = tmp_config['prob_remain'].tolist()
    ax.plot(x, y, color = color, linewidth = 2)
# title, 
custom_legend = [Line2D([0], [0], color = 'tab:blue', lw=4),
                 Line2D([0], [0], color = 'tab:orange', lw=4),
                 Line2D([0], [0], color = 'tab:red', lw=4)]
ax.legend(custom_legend, ['min', 'max', 'max(std)'])
plt.xticks(np.arange(0, 20, 1))
plt.suptitle('Examples', size = large_text)
plt.xlabel('Number of fixed traits', size = small_text)
plt.ylabel('Probability remain', size = small_text)
plt.savefig('../fig/COGSCI23/stability/case_study.pdf')

# what do they correspond to? 
entry_conf = pd.read_csv('../data/analysis/entry_configuration_master.csv')
config_min_sum = config_min_sum[['config_id']]
config_min_sum['type'] = 'min'
config_max_sum = config_max_sum[['config_id']]
config_max_sum['type'] = 'max'
config_max_std = config_max_std[['config_id']]
config_max_std['type'] = 'max(std)'
case_studies = pd.concat([config_min_sum, config_max_sum, config_max_std])
case_studies = entry_conf.merge(case_studies, on = 'config_id', how = 'inner')
case_studies
# min: Warrau (not complete)
# max(std): Warrau (not complete)
# max: Ancient Egypt, ... 