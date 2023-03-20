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

# get the files 
basepath = '../data/RCT/'
files = os.listdir(basepath)
files = [f for f in files if f.endswith('.csv')]

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

# formal burials ON grave goods 
d.groupby('intervention_var').size()
d.groupby('outcome_var').size()
d.groupby('trial').size()

burials_gravegoods = d[d['trial'] == '10_9']
y_population = configuration_probabilities[np.where(configurations[:, 8] == 1)[0]].sum()
fig, ax = plt.subplots()
sns.lineplot(data=burials_gravegoods,
             x='timestep',
             y='outcome',
             hue='identifier')
plt.ylabel('fraction with grave goods')
plt.hlines(y_population, 0, n_timesteps, color='tab:red', linestyles='dashed')
plt.suptitle('Formal Burials ON Grave Goods')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('../fig/intervention/burials_gravegoods.png', bbox_inches='tight')

# mind-body on formal burials 
afterlife_burials = d[d['trial'] == '5_10']
y_population = configuration_probabilities[np.where(configurations[:, 9] == 1)[0]].sum()
fig, ax = plt.subplots()
sns.lineplot(data=afterlife_burials, 
             x='timestep',
             y='outcome',
             hue='identifier') 
plt.ylabel('fraction with formal burials')
plt.suptitle('Afterlife Belief ON Formal Burials')
plt.hlines(y_population, 0, n_timesteps, color='tab:red', linestyles='dashed')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('../fig/intervention/afterlife_burials.png', bbox_inches='tight')

# opposite 
## this is BARELY significant
## so that is pretty interesting 
burials_afterlife = d[d['trial'] == '10_5']
y_population = configuration_probabilities[np.where(configurations[:, 4] == 1)[0]].sum()
fig, ax = plt.subplots()
sns.lineplot(data=burials_afterlife, 
             x='timestep',
             y='outcome',
             hue='identifier') 
plt.ylabel('fraction with afterlife belief')
plt.suptitle('Formal Burials ON Afterlife Belief')
plt.hlines(y_population, 0, n_timesteps, color='tab:red', linestyles='dashed')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('../fig/intervention/burials_afterlife.png', bbox_inches='tight')

#### which traits change the most in this population? ####
#### currently we have to run all of the experiments actually ####
#### because we focus on only Y=0 ####
#### of course we could look at how other traits change ####