import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import configuration as cn
from tqdm import tqdm 

# preprocessing 
from fun import bin_states 
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
n_nodes = 20
configurations = bin_states(n_nodes) 

# find probability for different attributes
probability_list = []
for i in range(20): 
    column_n = configurations[:, i]
    column_n_idx = np.where(column_n > 0)
    column_probs = configuration_probabilities[column_n_idx]
    mean_prob = np.mean(column_probs)
    std_prob = np.std(column_probs)
    probability_list.append((i+1, mean_prob, std_prob))
probability_df = pd.DataFrame(probability_list, columns = ['question_id', 'mean(prob)', 'std(prob)'])

# match with questions 
question_reference = pd.read_csv('../data/analysis/question_reference.csv')
question_reference = question_reference[['question_id', 'question']]
question_probability = question_reference.merge(probability_df, on = 'question_id', how = 'inner')
question_probability = question_probability.sort_values('mean(prob)').reset_index()

global_mean = np.mean(configuration_probabilities)

fig, ax = plt.subplots(dpi = 300)
for i, row in question_probability.iterrows(): 
    x = row['mean(prob)']
    x_err = row['std(prob)']
    plt.scatter(x, i, color = 'tab:blue')
plt.yticks(np.arange(0, 20, 1), question_probability['question'].values)
plt.vlines(global_mean, ymin = 0, ymax = 20, color = 'tab:red', ls = '--')
plt.xlabel('Mean probability')
plt.savefig('../fig/feature_stability.pdf', bbox_inches = 'tight')

## look at standard deviation (much larger than actual difference?) ##
# ...

# most enforced practices
d_enforcement = pd.read_csv('../data/COGSCI23/enforcement_observed.csv')
question_reference = pd.read_csv('../data/analysis/question_reference.csv')
observed_configs = d_enforcement['config_id'].unique().tolist()

# takes a couple of minutes
# not the most efficient approach
top_five_list = []
for config_idx in tqdm(observed_configs): 
    ConfObj = cn.Configuration(config_idx, 
                            configurations, 
                            configuration_probabilities)

    df = ConfObj.neighbor_probabilities(configurations,
                                        configuration_probabilities,
                                        question_reference,
                                        top_n = 5)

    df = df[['question_id', 'question']]
    df['config_id'] = config_idx 
    top_five_list.append(df)

top_five_df = pd.concat(top_five_list)
top_five_df = top_five_df.groupby('question').size().reset_index(name = 'count')
top_five_df = top_five_df.sort_values('count', ascending = True).reset_index()

# plot this 
fig, ax = plt.subplots(dpi = 300)
for i, row in top_five_df.iterrows(): 
    x = row['count']
    plt.scatter(x, i, color = 'tab:blue')
plt.yticks(np.arange(0, 20, 1), top_five_df['question'].values)
plt.xlabel('n(enforced first five)')
plt.savefig('../fig/number_enforced_first_five.pdf', bbox_inches = 'tight')

# tables
question_reference = pd.read_csv('../data/analysis/question_reference.csv')

### sacrifice ###
adult = configurations[:, 14]
child = configurations[:, 15]

adult_on = np.where(adult == 1)
adult_off = np.where(adult == -1)
child_on = np.where(child == 1)
child_off = np.where(child == -1)

## get the four quadrants
both_on = np.intersect1d(adult_on, child_on)
both_off = np.intersect1d(adult_off, child_off)
child_only = np.intersect1d(adult_off, child_on)
adult_only = np.intersect1d(adult_on, child_off)

## get probabilities
p_both_on = configuration_probabilities[both_on].mean()
p_both_off = configuration_probabilities[both_off].mean()
p_child_only = configuration_probabilities[child_only].mean()
p_adult_only = configuration_probabilities[adult_only].mean()

### big gods ###
monitor = configurations[:, 11]
punish = configurations[:, 12]

monitor_on = np.where(monitor == 1) 
monitor_off = np.where(monitor == -1)
punish_on = np.where(punish == 1)
punish_off = np.where(punish == -1)

## get the four quadrants 
both_on = np.intersect1d(monitor_on, punish_on)
both_off = np.intersect1d(monitor_off, punish_off)
monitor_only = np.intersect1d(monitor_on, punish_off)
punish_only = np.intersect1d(monitor_off, punish_on)

## get probabilities
p_both_on = configuration_probabilities[both_on].mean()
p_both_off = configuration_probabilities[both_off].mean()
p_monitor_only = configuration_probabilities[monitor_only].mean()
p_punish_only = configuration_probabilities[punish_only].mean()

p_both_on
p_both_off
p_monitor_only
p_punish_only

# Dynamics: child sacrifice and adult sacrifice 
adult = 14
child = 15

x_configurations = np.delete(configurations, [14, 15], 1)
x_configurations = np.unique(x_configurations, axis = 0)

np.random.seed(seed=1)
idx = len(x_configurations)
sample_idx = np.random.choice(idx,
                              size = 1000, 
                              replace = False)
sample_configs = x_configurations[[sample_idx]]
transition_probabilities = []
for num, x in tqdm(enumerate(sample_configs)): 
    # get the configurations 
    conf_both = np.insert(x, adult, [1, 1])
    conf_none = np.insert(x, adult, [-1, -1])
    conf_cs = np.insert(x, adult, [-1, 1])
    conf_as = np.insert(x, adult, [1, -1])
    # get the configuration ids 
    idx_both = np.where(np.all(configurations == conf_both, axis = 1))[0][0]
    idx_none = np.where(np.all(configurations == conf_none, axis = 1))[0][0]
    idx_cs = np.where(np.all(configurations == conf_cs, axis = 1))[0][0]
    idx_as = np.where(np.all(configurations == conf_as, axis = 1))[0][0]
    # get probabilities
    p_both = configuration_probabilities[idx_both]
    p_none = configuration_probabilities[idx_none]
    p_cs = configuration_probabilities[idx_cs]
    p_as = configuration_probabilities[idx_as]
    # put this together
    for p_focal, type_focal in zip([p_both, p_none, p_cs, p_as], ['CS, AS', '~CS, ~AS', 'CS, ~AS', '~CS, AS']): 
        if type_focal == 'CS, AS' or type_focal == '~CS, ~AS': 
            p_neighbors = [p_cs, p_as]
            type_neighbors = ['CS, ~AS', '~CS, AS'] # none = (~CS, ~AS)
        else: 
            p_neighbors = [p_both, p_none]
            type_neighbors = ['CS, AS', '~CS, ~AS']
        for p_neighbor, type_neighbor in zip(p_neighbors, type_neighbors): 
            flow = p_neighbor / (p_focal + sum(p_neighbors))
            transition_probabilities.append((num, type_focal, type_neighbor, flow))


x = [(x, y, z) for a, x, y, z in transition_probabilities]
df = pd.DataFrame(x, columns = ['type_from', 'type_to', 'probability'])
df = df.groupby(['type_from', 'type_to'])['probability'].mean().reset_index(name = 'probability')

# make a plot 
import networkx as nx 
G = nx.from_pandas_edgelist(df,
                            source = 'type_from',
                            target = 'type_to',
                            edge_attr = 'probability',
                            create_using = nx.DiGraph)


edge_width = []
edge_labels = {}

for x, y, attr in G.edges(data = True): 
    weight = attr['probability']
    edge_width.append(weight)
    edge_labels[(x, y)] = round(weight, 2)


G.edges(data=True)
edge_labels

pos = {'CS, AS': (2, 2),
       '~CS, ~AS': (2, 0),
       '~CS, AS': (0, 1),
       'CS, ~AS': (4, 1)}

node_labels = {}
for i in G.nodes(): 
    node_labels[i] = i
node_labels

### FIX!! 
label_pos = [(-0.2, -0.2, 3, 1.5, '0.22'), # (CS, AS) -> (CS, ~AS) [x] []
             (-0.65, 0.65, 1, 1.5, '0.23'), # (CS, AS) -> (~CS, AS) [x]
             (0.1, 0.1, 3, 1.5, '0.4'), # (CS, ~AS) -> (CS, AS) [x]
             (-0.65, 0.65, 3, 0.5, '0.53'), # (CS, ~AS) -> (~CS, ~AS) [x]
             (0, 0, 1, 1.5, '0.4'), # (~CS, AS) -> (CS, AS) [x]
             (-0.2, -0.2, 1, 0.5, '0.53'), # (~CS, AS) -> (~CS, ~AS) [x]
             (0, 0, 3, 0.5, '0.16'), # (~CS, ~AS) -> (CS, ~AS) [x]
             (0.05, 0.05, 1, 0.5, '0.19')] # (~CS, ~AS) -> (~CS, AS) [x]

fig, ax = plt.subplots(dpi = 300, figsize = (8, 8))
nx.draw_networkx_nodes(G, pos, node_size = 4500, linewidths = 2,
                       edgecolors = 'k',
                       node_color = 'white')
nx.draw_networkx_edges(G, pos, width = [x*5 for x in edge_width],
                       connectionstyle = "arc3,rad=0.2",
                       node_size = 5500)
nx.draw_networkx_labels(G, pos, node_labels)

for num, ele in enumerate(label_pos): 
    x_add, y_add, x, y, text = ele
    if num in [0, 2, 5, 7]:
        ax.text(x = x+x_add, y = y+y_add, s = text, 
                rotation = -50, 
                fontsize = 20, 
                color = 'black')
    else:
        ax.text(x = x+x_add, y = y, s = text, 
            rotation = 50, 
            fontsize = 20, 
            color = 'black')
        
plt.show();


# find monitor, but not punish 
pd.set_option('display.max_colwidth', None)
d = pd.read_csv('/home/vmp/humanities-glass/data/analysis/entry_maxlikelihood.csv')
unique_config_idx = d['config_id'].unique().tolist()
unique_configs = configurations[[unique_config_idx]]

monitor = np.where(unique_configs[:, 11] == -1) 
punish = np.where(unique_configs[:, 12] == 1)

obs_cases = np.intersect1d(monitor, punish)
unique_config_idx = np.array(unique_config_idx)
cases = unique_config_idx[obs_cases]

d_case = d[d['config_id'].isin(cases)]

# double check one 
config_id = 362112
ConfObj = cn.Configuration(config_id,
                           configurations,
                           configuration_probabilities)

question_reference = pd.read_csv('../data/analysis/question_reference.csv')
question_reference['sanity_check'] = ConfObj.configuration
