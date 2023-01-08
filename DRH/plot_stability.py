import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# plotting setup
small_text = 12
large_text = 18

# import 
stability = pd.read_csv('../data/COGSCI23/evo_stability/maxlik_evo_stability.csv')

# plot 
sns.scatterplot(data = stability, 
                x = 'config_prob',
                y = 'remain_prob')
plt.xlabel('p(configuration)', size = small_text)
plt.ylabel('p(remain)', size = small_text)

# we could fit a model 
# to this if we wanted to 
# clearly not linear 

# annotations 
## find some of the outliers
min_config = stability.sort_values('config_prob').head(1)
max_config = stability.sort_values('config_prob').tail(1)
min_remain = stability.sort_values('remain_prob').head(1)
max_remain = stability.sort_values('remain_prob').tail(1)
## a bit more specific outliers
outlier_top1 = stability[(stability['remain_prob'] > 0.86) & 
                         (stability['config_prob'] < 0.002)]
outlier_top2 = stability[(stability['remain_prob'] > 0.85) &
                         (stability['config_prob'] < 0.004) &
                         (stability['config_prob'] > 0.003)]
outlier_bot1 = stability[(stability['remain_prob'] < 0.8) &
                         (stability['remain_prob'] > 0.79) &
                         (stability['config_prob'] > 0.002)]
outlier_bot2 = stability[(stability['remain_prob'] < 0.82) &
                         (stability['remain_prob'] > 0.81) & 
                         (stability['config_prob'] > 0.0035)]
## gather all of them 
annotations = pd.concat([min_config, max_config, min_remain,
                         max_remain, outlier_top1, outlier_top2,
                         outlier_bot1, outlier_bot2])
annotations = annotations.drop_duplicates()
## now find the corresponding religions 
pd.set_option('display.max_colwidth', None)
entry_configuration = pd.read_csv('../data/analysis/entry_configuration_master.csv')
entry_configuration = entry_configuration[['config_id', 'entry_drh']].drop_duplicates()
entry_configuration = entry_configuration.groupby('config_id')['entry_drh'].unique().reset_index(name = 'entry_drh')
annotations = entry_configuration.merge(annotations, on = 'config_id', how = 'inner')
annotations = annotations.sort_values('config_id')
## short names for the entries 
entries = pd.DataFrame({
    'config_id': [362374, 501634, 503686, 
                 769975, 929282, 1017735,
                 1025926, 1036215],
    'entry_short': ['Pauline', 'PRC Catholics', 'Donatism',
                   'Aztec', 'Sadducees', 'Nestorian',
                   'Cistercians', 'pre-Christian Irish']})
## merge back in 
annotations = annotations.merge(entries, on = 'config_id', how = 'inner')
annotations = annotations.drop(columns = {'entry_drh'})

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

# plot
fig, ax = plt.subplots(dpi = 300)
## the scatter
sns.scatterplot(data = stability, 
                x = 'config_prob',
                y = 'remain_prob',
                c = stability['color'].values)
## the annotations 
for _, row in annotations.iterrows(): 
    x = row['config_prob']
    y = row['remain_prob']
    label = row['entry_short']
    # Cistercians break the plot currently
    if label == 'Cistercians': 
        ax.annotate(label, xy = (x-0.0003, y),
                    horizontalalignment = 'right',
                    verticalalignment = 'center')
    else: 
        ax.annotate(label, xy = (x+0.0003, y),
                    horizontalalignment = 'left',
                    verticalalignment = 'center')
plt.xlabel('p(configuration)', size = small_text)
plt.ylabel('p(remain)', size = small_text)