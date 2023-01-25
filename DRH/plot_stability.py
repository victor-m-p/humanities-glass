'''
VMP 2023-01-08: 
updated and run on new parameter file.
produces scatterplot of log(P(configuration)) x p(remain)
'''

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

# plotting setup
small_text = 12
large_text = 18

# import 
stability = pd.read_csv('../data/COGSCI23/evo_stability/maxlik_evo_stability.csv')

## to log 
stability['log_config_prob'] = np.log(stability['config_prob'])

# annotations 
## above trendline 
aztec = stability[(stability['remain_prob'] > 0.85) & 
                  (stability['log_config_prob'] < -6.5) &
                  (stability['log_config_prob'] > -7.5)]
pauline = stability[(stability['remain_prob'] > 0.89) &
                    (stability['log_config_prob'] > -4.5)]
top_1 = stability[(stability['remain_prob'] > 0.75) &
                  (stability['log_config_prob'] < -11)]
top_2 = stability[(stability['remain_prob'] > 0.79) &
                  (stability['log_config_prob'] > -10) &
                  (stability['log_config_prob'] < -9.5)]
top_3 = stability[(stability['remain_prob'] > 0.83) &
                  (stability['log_config_prob'] > -8.5) &
                  (stability['log_config_prob'] < -8)]
## below trendline
sadducees = stability[(stability['remain_prob'] < 0.69) &
                  (stability['log_config_prob'] < -13)]
bot_1 = stability[(stability['remain_prob'] < 0.69) & 
                  (stability['log_config_prob'] > -12) & 
                  (stability['log_config_prob'] < -10.5)]
bot_2 = stability[(stability['remain_prob'] > 0.72) & 
                  (stability['remain_prob'] < 0.73) & 
                  (stability['log_config_prob'] < -8) &
                  (stability['log_config_prob'] > -8.5)]
bot_3 = stability[(stability['remain_prob'] > 0.74) & 
                  (stability['remain_prob'] < 0.76) & 
                  (stability['log_config_prob'] < -7) &
                  (stability['log_config_prob'] > -7.5)]
bot_4 = stability[(stability['remain_prob'] > 0.8) & 
                  (stability['remain_prob'] < 0.84) & 
                  (stability['log_config_prob'] < -4.5) & 
                  (stability['log_config_prob'] > -5.2)]

## gather all of them 
annotations = pd.concat([aztec, pauline, top_1, top_2, top_3,
                         sadducees, bot_1, bot_2, bot_3, bot_4])
annotations = annotations.drop_duplicates()

## now find the corresponding religions 
pd.set_option('display.max_colwidth', None)
entry_configuration = pd.read_csv('../data/analysis/entry_configuration_master.csv')
entry_configuration = entry_configuration[['config_id', 'entry_name']].drop_duplicates()
entry_configuration = entry_configuration.groupby('config_id')['entry_name'].unique().reset_index(name = 'entry_name')
annotations = entry_configuration.merge(annotations, on = 'config_id', how = 'inner')
annotations = annotations.sort_values('config_id')

## short names for the entries 
entries = pd.DataFrame({
    'config_id': [362374, 
                  370630,
                  372610, 501638, #525313,
                  #652162, 
                  769975, 774016,
                  #894854, 
                  896898, 913282,
                  929282, 978831, 995207,
                  #1016839
                  ],
    'entry_short': ['Pauline', #'Marcionites'
                    'Muslim Students US/CA',
                    'Yolngu', 'Donatism', #'Soviet Atheism',
                    #'Santal', 
                    'Aztec', 'Pagans under Julian',
                    #'Circumcellions', 
                    'Iban', 'Rwala Bedouin',
                    'Sadducees', 'Tang Tantrism', 'Samaritans',
                    #'Muridiyya Senegal'
                    ]})
## merge back in 
annotations = annotations.merge(entries, on = 'config_id', how = 'inner')
annotations = annotations.drop(columns = {'entry_name'})

## prepare colors 
annotations_id = annotations[['config_id']]
stability = stability.merge(annotations_id, 
                            on = 'config_id',
                            how = 'left',
                            indicator = True)
stability = stability.rename(columns = {'_merge': 'color'})
stability = stability.replace({'color': {'left_only': 'tab:blue',
                                         'both': 'tab:orange'}})
stability = stability.sort_values('color')

# median config
median_config = stability['log_config_prob'].median()

# plot
fig, ax = plt.subplots(dpi = 300)

## the scatter
sns.scatterplot(data = stability, 
                x = 'log_config_prob',
                y = 'remain_prob',
                c = stability['color'].values)
sns.regplot(data = stability,
           x = 'log_config_prob',
           y = 'remain_prob',
           scatter = False, 
           color = 'tab:red')
## the annotations 
plt.axvline(x = median_config,
           ymin = 0,
           ymax = 1,
           ls = '--',
           color = 'tab:red')
for _, row in annotations.iterrows(): 
    x = row['log_config_prob']
    y = row['remain_prob']
    label = row['entry_short']
    # specifics 
    if label == 'Samaritans': 
        ax.annotate(label, xy = (x-0.1, y+0.01),
                    horizontalalignment = 'right',
                    verticalalignment = 'center')
    elif label in ['Pauline', 'Astec', 
                 'Muslim Students US/CA', 'Muridiyya Senegal',
                 'Tang Tantrism', 'Soviet Atheism']: 
        ax.annotate(label, xy = (x-0.1, y),
                    horizontalalignment = 'right',
                    verticalalignment = 'center')
    else: 
        ax.annotate(label, xy = (x+0.1, y),
                    horizontalalignment = 'left',
                    verticalalignment = 'center')
plt.xlabel('Log(P(configuration))', size = small_text)
plt.ylabel('P(remain)', size = small_text)
plt.xlim(-14, -3.6)
## save figure 
plt.savefig('../fig/stability.pdf')

''' create groups in the data '''

## regression line  
from sklearn.linear_model import LinearRegression
log_config_prob = stability['log_config_prob'].values.reshape(-1, 1)
remain_prob = stability['remain_prob'].values.reshape(-1, 1)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(log_config_prob, remain_prob)  # perform linear regression
Y_pred = linear_regressor.predict(log_config_prob)  # make predictions

# color plot by this to be sure: 
stability['pred_remain'] = Y_pred
stability['above_line']= np.where((stability['pred_remain']-stability['remain_prob']) > 0, 1, 0)

## above median config?
stability['above_median'] = [1 if x>median_config else 0 for x in stability['log_config_prob']]

## four combinations ## 
#define conditions
conditions = [
    (stability['above_line'] == 1) & (stability['above_median'] == 1),
    (stability['above_line'] == 1) & (stability['above_median'] == 0),
    (stability['above_line'] == 0) & (stability['above_median'] == 1),
    (stability['above_line'] == 0) & (stability['above_median'] == 0)
]

#define results
results = ['Global Peak', 'Local Peak', 'Mountain Range', 'Valley']

#create new column based on conditions in column1 and column2
stability['Landscape'] = np.select(conditions, results)

## take out useful columns 
stability = stability[['config_id', 'config_prob', 'Landscape']]

## 
entry_maxlikelihood = pd.read_csv('../data/analysis/entry_maxlikelihood.csv')
entry_maxlikelihood = entry_maxlikelihood[['config_id', 'entry_name']]
stability = stability.merge(entry_maxlikelihood, on = 'config_id', how = 'inner')

## sort by stuff 
stability = stability.sort_values(by = ['Landscape', 'config_prob'],
                                  ascending = [True, False])

stability['config_prob'] = [round(x*100, 2) for x in stability['config_prob']]

## rename stuff
stability = stability.rename(columns = {'config_id': 'Configuration',
                                        'config_prob': 'P(configuration)',
                                        'entry_name': 'Entry Name'})

# to latex table
stability_latex = stability.to_latex(index=False)
with open('../tables/landscape_types.txt', 'w') as f: 
    f.write(stability_latex)

    
''' create overview table '''
# helper function
def community_weight(d, 
                     sort_column,
                     configuration_id,
                     configuration_prob):
    config_dict = {}
    weight_dict = {}
    for comm in d[sort_column].unique(): 
        config_list = []
        weight_list = []
        network_comm = d[d[sort_column] == comm]
        for _, row in network_comm.iterrows():
            config_id = int(row[configuration_id])
            config_prob = row[configuration_prob]
            CommObj = cn.Configuration(config_id, 
                                       configurations,
                                       configuration_probabilities)
            conf = CommObj.configuration
            config_list.append(conf)
            weight_list.append(config_prob)
        config_dict[comm] = config_list 
        weight_dict[comm] = weight_list
    return config_dict, weight_dict 

# get the proper weighting 
def clade_wrangling(c, w, question_reference):

    # get values out 
    c1, w1 = c.get(0), w.get(0)
    c2, w2 = c.get(1), w.get(1)
    # stack
    s1, s2 = np.stack(c1, axis = 1), np.stack(c2, axis = 1)
    # recode
    s1[s1 == -1] = 0
    s2[s2 == -1] = 0
    # weights
    wn1, wn2 = np.array(w1)/sum(w1), np.array(w2)/sum(w2)
    # average
    bit1 = np.average(s1, axis = 1, weights = wn1)
    bit2 = np.average(s2, axis = 1, weights = wn2)
    # turn this into dataframes
    df = pd.DataFrame(
        {f'Focal': bit1,
         f'Other': bit2})
    df['question_id'] = df.index + 1
    # merge with question reference
    df = df.merge(question_reference, on = 'question_id', how = 'inner')
    # difference 
    df = df.assign(absolute_difference = lambda x: 
        np.abs(x['Focal']-x['Other']))
    return df 

def subset_groups(df, sub_list, remap_dict): 
    df = df[df['Landscape'].isin(sub_list)]
    df['Focal'] = [remap_dict.get(x) for x in df['Landscape']]
    return df

# ... 
stability = stability[['Configuration', 'P(configuration)', 'Landscape']].drop_duplicates()

import configuration as cn 
question_reference = pd.read_csv('../data/analysis/question_reference.csv')

# preprocessing 
from fun import bin_states 
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
n_nodes = 20
configurations = bin_states(n_nodes) 

### each community against the others ### 
n_groups = 4
subset_list = ['Global Peak', 'Local Peak', 'Mountain Range', 'Valley']
A = np.zeros((n_groups, n_groups), int)
np.fill_diagonal(A, 1)

clade_list = []
for row in A: 
    dct = {group:val for group, val in zip(subset_list, row)}
    df = subset_groups(stability, subset_list, dct)
    cdict, wdict = community_weight(df, 
                                    'Focal',
                                    'Configuration',
                                    'P(configuration)')
    clade = clade_wrangling(cdict, wdict, question_reference)
    clade_list.append(clade)

def wrangle_subset(landscape_list, landscape_n, landscape_name, n_top):
    subset = landscape_list[landscape_n] # Global Peak
    subset['Landscape'] = landscape_name
    subset = subset.sort_values('absolute_difference', ascending = False)
    subset = subset.head(n_top)
    return subset 

global_peak = wrangle_subset(clade_list, 0, 'Global Peak', 5)
local_peak = wrangle_subset(clade_list, 1, 'Local Peak', 5)
mountain_range = wrangle_subset(clade_list, 2, 'Mountain Range', 5)
valley = wrangle_subset(clade_list, 3, 'Valley', 5)

landscapes = pd.concat([global_peak, local_peak, 
                        mountain_range, valley])

landscapes = landscapes.assign(focal_minus_other = lambda x: (x['Focal']-x['Other'])*100)

# Global Peak: 
## 20.39: distinct written language
## 16.12: Co-sacrifices
## -14.70: Large-scale rituals
## -13.52: Formal Burials 

# Local Peak
## 29.01: Large-Scale 
## 27.18: Supernatural monitoring
## -24.38: Reincarnation this world
## 22.45: Supernatural punishment

# Mountain Range
## -18.45: Distinct Written 
## -14.00: Co-sacrifices
## 10.41: Reincarnation this world
## 8.45: Special treatment for corpses

# Valley: 
## 36.69: Supernatural punish
## 34.63: Supernatural monitor
## 24.25: Large-scale rituals
## -23.45: Co-sacrifices 
