import numpy as np 
import pandas as pd 
from fun import bin_states, hamming_distance, top_n_idx
from sklearn.manifold import MDS
import itertools
import networkx as nx 
import matplotlib.pyplot as plt
import os 
from matplotlib.patches import Ellipse

# setup 
n_nodes, n_nan = 20, 5
seed = 254
n_cutoff = 500
outpath = '../fig'
start_path = '../data/analysis/'
end_path = f'_nrows_455_maxna_{n_nan}_nodes_{n_nodes}'

# loads
probabilities = np.loadtxt(f'{start_path}p{end_path}.txt') # used 
pos = np.loadtxt(f'{start_path}pos{end_path}_cutoff_{n_cutoff}_seed_{seed}.txt') # used 
#allstates = np.loadtxt(f'{start_path}allstates{end_path}.txt')  
d_likelihood = pd.read_csv(f'{start_path}d_likelihood{end_path}.csv')
mat_likelihood = np.loadtxt(f'{start_path}mat_likelihood{end_path}.txt')
nodes_reference = pd.read_csv(f'{start_path}nref{end_path}.csv')

# curate node information
## get p indices and p values for the top 500 data states
p_ind, p_vals = top_n_idx(n_cutoff, probabilities) # same n_cutoff as prep_general.py
d_ind = pd.DataFrame({'p_ind': p_ind, 'config_w': p_vals})
d_ind['node_id'] = d_ind.index

## add entry_id information to likelihood dataframe
d_likelihood = d_likelihood.merge(nodes_reference, on = 'entry_id', how = 'inner')

## preserve dtypes (integers) through merge
d_ind = d_ind.convert_dtypes()
d_likelihood = d_likelihood.convert_dtypes()

## merge 
d_nodes = d_likelihood.merge(d_ind, on = 'p_ind', how = 'outer', indicator = True)
d_nodes.rename(columns = {'_merge': 'state'}, inplace = True)

## only states that are in both for some types of information
d_conf = d_nodes[d_nodes['state'] == 'both']

### observation weight (added for normalization of weight)
d_obsw = d_conf.groupby('entry_id').size().reset_index(name = 'entry_count')
d_obsw = d_obsw.assign(entry_weight = lambda x: 1/x['entry_count'])
d_conf = d_conf.merge(d_obsw, on = 'entry_id', how = 'inner')

### weight for configurations (as observed in data states)
node_sum = d_conf.groupby('node_id')['entry_weight'].sum().reset_index(name = 'config_sum')

## maximum likelihood (ML) data states by configuration
#### here we should manually choose rather than sample randomly.
#### this is also where we loose reproducibility. 
node_ml = d_conf.groupby(['node_id'])['p_norm'].max().reset_index(name = 'p_norm')
node_ml = node_ml.merge(d_conf, on = ['node_id', 'p_norm'], how = 'inner')

##### here we make a qualitative / story-telling choice 
node_decision = node_ml.groupby('node_id').size().reset_index(name = 'count').sort_values('count', ascending=False)
node_decision = node_decision[node_decision['count'] > 1]
node_decision = node_decision.merge(node_ml, on = 'node_id', how = 'inner')
node_decision = node_decision[['node_id', 'entry_name', 'entry_id']]
unique_states = node_decision['node_id'].unique()

##### page through all unique states and choose
node_decision[node_decision['node_id'] == unique_states[21]]
ml_choices = pd.DataFrame({
    'node_id': [427, 455, 78, 183, 81, 464, 444, 
                421, 423, 98, 465, 76, 228, 371,
                439, 350, 319, 298, 176, 395, 163, 166],
    'entry_id': [654, 738, 879, 483, 1311, 415, 1140, 
                 839, 1076, 484, 926, 842, 891, 921,
                 1419, 984, 1129, 217, 486, 1374, 1127, 564]
})

node_ml = node_ml.merge(ml_choices, on = ['node_id', 'entry_id'], how = 'left', indicator = True)
node_ml = node_ml.sort_values(['node_id', '_merge'], ascending=False).groupby('node_id').head(1)
node_ml = node_ml[['node_id', 'entry_id', 'entry_name']]

## weight of each configuration 
node_overlap = node_sum.merge(node_ml, on = 'node_id', how = 'inner')
d_all = d_nodes[(d_nodes['state'] == 'right_only') | (d_nodes['state'] == 'both')]
d_all = d_all[['node_id', 'config_w']].drop_duplicates()
node_attr = d_all.merge(node_overlap, on = 'node_id', how = 'left', indicator = True) # should have len = n (n = 500)
node_attr = node_attr.rename(columns = {'_merge': 'datastate'})
node_attr.replace({'datastate': {'both': 'Yes', 'left_only': 'No'}},
                  inplace = True)
node_attr = node_attr.sort_values('node_id', ascending=True).reset_index(drop=True)

# couple configurations, data states and ids
comb = list(itertools.combinations(node_attr['node_id'].values, 2))

# create network
G = nx.Graph(comb)
node_attr
dct_nodes = node_attr.to_dict('index')
for key, val in dct_nodes.items():
    G.nodes[key]['node_id'] = val['node_id']
    G.nodes[key]['size'] = val['config_w']
    G.nodes[key]['datastate'] = val['datastate']
    G.nodes[key]['entry_id'] = val['entry_id']
    G.nodes[key]['entry_name'] = val['entry_name']
    G.nodes[key]['ds_weight'] = val['config_sum']

# create one great plot 

## prepare main plot 
node_scaling = 3000
node_size_lst = list(nx.get_node_attributes(G, 'size').values())
node_size_lst = [x*node_scaling for x in node_size_lst]

## prepare sub plot 
### get ids of data states
sub_ids = list(nx.get_node_attributes(G, 'node_id').values())
sub_vals = list(nx.get_node_attributes(G, 'datastate').values())
sub_nodelst = [id if val == 'Yes' else np.nan for id, val in zip(sub_ids, sub_vals)]

### get weight of states 
sub_size_lst = list(nx.get_node_attributes(G, 'ds_weight').values())

## clean them 
subnode_scaling = 5
sub_nodelst = [x for x in sub_nodelst if ~np.isnan(x)]
sub_size_lst = [x for x in sub_size_lst if ~np.isnan(x)]
sub_size_lst = [x*subnode_scaling for x in sub_size_lst]

## annotations
### find interesting things to annotate 
### tomorrow ...

### top 10 most plausible civilizations
n_annotate = 10
node_attr_yes = node_attr[node_attr['datastate'] == 'Yes']
df_top_n = node_attr_yes.sort_values('config_w', ascending=False)

focus_nodes = sorted([427, 444, 81, 428, 494, 183, 78, 
                228, 421, 79, 123, 60, 208,
                84, 11, 187, 76, 88, 425, 392, 
                106, 312, 473, 98])

## find particular nodes
sums = [np.sum([x, y]) for x, y in pos]
dsum = pd.DataFrame({'sums': sums})
dsum['node_id'] = dsum.index 
dsum = dsum.merge(node_attr, on = 'node_id', how = 'inner')
dsum.sort_values('sums', ascending=True)

# make better translations
transl = {
    11: 'Spiritualism',
    44: 'Tiwi',
    60: 'Peyote',
    76: 'US Evangelical',
    78: 'Free Methodist',
    79: 'Sourthern Baptist',
    81: 'Jehovah',
    #84: 'Montanism',
    #88: 'Pauline Christianity',
    98: 'Northern Irish Protestants',
    106: 'Beat Buddhism',
    #123: 'Modern Mystery School',
    183: 'Northern Irish Catholics',
    187: 'Society of Jesus',
    208: 'US Hinduism',
    228: 'Peruvian Mormons',
    281: 'Orokaiva',
    312: 'Spartan Cults',
    #392: 'Irish Catholicism',
    421: 'German Protestantism',
    425: 'Roman Orthodox',
    427: 'Cistercians',
    428: 'Islam in Aceh',
    444: 'Medieval Confusianism',
    473: 'Indian Buddhism',
    494: 'Pre-Christian Ireland', 
}

nudge = {
    11: (-4, -4), #spiritualism
    44: (-4, -2), # Tiwi
    60: (-2, -6), #peyote
    76: (4, -3), #US evangelical
    78: (5, -3), #methodist
    79: (4, -1), # southern baptist
    81: (5, 0.5), #jehovah
    84: (0, 0), #montanism
    88: (0, 0), #paul
    98: (5, 0.5), #north irish prot.
    106: (0, -5), #beat
    123: (-2, 0), #mystery
    183: (5, 1), #north irish cat.
    187: (-12, 0), #society of jesus
    208: (3, -1.5), # US Hinduism
    228: (5, 1), #Peruvian Mormons
    281: (-4, -2), #Orokaiva
    312: (-8, 0), #Spartan Cults
    392: (5, 0.5), #Irish Cat
    421: (-13, -1.5), #German Prot.
    425: (-12, 3), #Roman Ort.
    427: (5, 1.5), #Cistercians
    428: (5, 1.5), #Islam Aceh
    444: (-12, 3), #Medieval Conf
    473: (-12, 0), #Indian Budh.
    494: (5, 2)} # Pre-christian Ireland

labeldct = {}
positions = []
for key, val in dct_nodes.items(): 
    if key in transl.keys(): 
        entry_name = val['entry_name']
        position = pos[key]
        labeldct[key] = entry_name
        positions.append((entry_name, position, key))

# paths and names
perc = sum(p_vals)
out = os.path.join(outpath, f'MDS_annotated_nnodes_{n_nodes}_maxna_{n_nan}_ncutoff_{n_cutoff}_perc_{perc}_seed_{seed}.pdf')

#### actually plotting ####
fig, ax = plt.subplots(facecolor = 'w', edgecolor = 'k', dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size = node_size_lst, node_color = 'tab:blue')
nx.draw_networkx_nodes(G, pos, nodelist = sub_nodelst, node_size = sub_size_lst,
                       node_color = 'tab:orange', node_shape = 'x',
                       linewidths=0.4)

# annotate 
for entry_name, position, node_id in positions: 
    pos_x, pos_y = position 
    abb = transl.get(node_id)
    x_nudge, y_nudge = nudge.get(node_id)
    ax.annotate(abb, xy=[pos_x, pos_y], 
                xytext=[pos_x+x_nudge, pos_y+y_nudge],
                arrowprops = dict(arrowstyle="->",
                                    connectionstyle="arc3"))

plt.savefig(out)

# drawing ellipses
## red ellipse
ellipse = Ellipse((-2, 2.5),
    width=7.5, # tilt 
    height=4.5,
    angle=58,
    facecolor='tab:blue',
    edgecolor='tab:blue',
    alpha=0.2)
ax.add_patch(ellipse)

## green ellipse
ellipse = Ellipse((7.7, -10.1),
    width=13, 
    height=3,
    angle=42,
    facecolor='tab:green',
    edgecolor='tab:green',
    alpha=0.2)
ax.add_patch(ellipse)

## the top one 
ellipse = Ellipse((3, -0.5),
    width=10, 
    height=4,
    angle=-60,
    facecolor='tab:red',
    edgecolor='tab:red',
    alpha=0.2)
ax.add_patch(ellipse)

## the bottom one 
ellipse = Ellipse((-0.4, -3.5),
    width=11, 
    height=2.5,
    angle=-48,
    facecolor='tab:red',
    edgecolor='tab:red',
    alpha=0.2)
ax.add_patch(ellipse)

nx.draw_networkx_nodes(G, pos, node_size = node_size_lst, node_color = 'tab:blue')
nx.draw_networkx_nodes(G, pos, nodelist = sub_nodelst, node_size = sub_size_lst,
                       node_color = 'tab:orange', node_shape = 'x',
                       linewidths=0.4)

# annotate 
for entry_name, position in positions: 
    pos_x, pos_y = position
    abb = translation_dct.get(entry_name)
    x_nudge, y_nudge = position_nudge.get(abb)
    ax.annotate(abb, xy=[pos_x, pos_y], 
                xytext=[pos_x+x_nudge, pos_y+y_nudge],
                arrowprops = dict(arrowstyle="->",
                                    connectionstyle="arc3"))

# save
plt.savefig(out)
translation_dct

node_attr[node_attr['node_id'] == 146]
# double check everything.
# some inconsistency. 
# rerun (nb. incorporate in earlier pipeline for final)
# more entry_name duplicates consider.

#### check which points are inside ellipse ####
# https://stackoverflow.com/questions/37031356/check-if-points-are-inside-ellipse-faster-than-contains-point-method
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig,ax = plt.subplots(1)
ax.set_aspect('equal')

# Some test points
x = np.random.rand(500)*0.5+0.7
y = np.random.rand(500)*0.5+0.7

# The ellipse
g_ell_center = (0.8882, 0.8882)
g_ell_width = 0.36401857095483
g_ell_height = 0.16928136341606
angle = 30.

g_ellipse = patches.Ellipse(g_ell_center, g_ell_width, g_ell_height, angle=angle, fill=False, edgecolor='green', linewidth=2)
ax.add_patch(g_ellipse)
cos_angle = np.cos(np.radians(180.-angle))
sin_angle = np.sin(np.radians(180.-angle))

xc = x - g_ell_center[0]
yc = y - g_ell_center[1]

xct = xc * cos_angle - yc * sin_angle
yct = xc * sin_angle + yc * cos_angle 

rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)

# Set the colors. Black if outside the ellipse, green if inside
colors_array = np.array(['black'] * len(rad_cc))
colors_array[np.where(rad_cc <= 1.)[0]] = 'green'

ax.scatter(x,y,c=colors_array,linewidths=0.3)

plt.show()
 