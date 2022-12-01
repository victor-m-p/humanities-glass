import numpy as np 
import pandas as pd 
from fun import bin_states, compute_HammingDistance, top_n_idx
from sklearn.manifold import MDS
import itertools
import networkx as nx 
import matplotlib.pyplot as plt
import os 
from matplotlib.patches import Ellipse

# setup 
n_nodes, maxna = 20, 10
seed = 2
n_cutoff = 500
outpath = '../fig'

# read files 
# consider what we actually need, pre-running POS now 
p_file = '../data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt'
mat_file = '../data/clean/matrix_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.txt'
d_main = '../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
sref = '../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
nref = '../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
p = np.loadtxt(p_file)
d_main = pd.read_csv(d_main)
sref = pd.read_csv(sref)
nref = pd.read_csv(nref)
d_likelihood = pd.read_csv('../data/analysis/d_likelihood_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.csv')
mat_likelihood = np.loadtxt('../data/analysis/mat_likelihood_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt').astype(int)

# get all state configurations 
allstates = bin_states(n_nodes) # takes a minute (do not attempt with n_nodes > 20)

# subset states above threshold (n = 500)
#def top_n_idx(n, p, allstates): # fix this
#    val_cutoff = np.sort(p)[::-1][n]
#    p_ind = [i for i, v in enumerate(p) if v > val_cutoff]
#    p_vals = p[p > val_cutoff]
#    return p_ind, p_vals
c_cutoff = 500
p_ind, p_vals = top_n_idx(500, p, allstates) 

# find out which of these substates is also a possible
# combination of the data that we have - 
# we want to have it weighted in this case 
d = pd.read_csv('../data/analysis/d_likelihood_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.csv')
mat = np.loadtxt('../data/analysis/mat_likelihood_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt').astype(int)
d = d.merge(sref, on = 'entry_id', how = 'inner')

# merge data
d_ind = pd.DataFrame({'p_ind': p_ind, 'config_w': p_vals}) # this needs double checking
d_ind['node_id'] = d_ind.index
d_ind = d_ind.convert_dtypes() # preserve int
d = d.convert_dtypes() # preserve int
d_merge = d.merge(d_ind, on = 'p_ind', how = 'outer', indicator = True)

# only stuff that appears in the data
# and that we have data for 
d_conf = d_merge[d_merge['_merge'] == 'both']

# curate different types of information
## observation weight (added for normalization of weight)
d_obsw = d_conf.groupby('entry_id').size().reset_index(name = 'entry_count')
d_obsw = d_obsw.assign(entry_weight = lambda x: 1/x['entry_count'])
d_conf = d_conf.merge(d_obsw, on = 'entry_id', how = 'inner')

## weight for configurations (as observed in data states)
node_sum = d_conf.groupby('node_id')['entry_weight'].sum().reset_index(name = 'config_sum')

## maximum likelihood (ML) data states by configuration
node_ml = d_conf.groupby(['node_id'])['p_norm'].max().reset_index(name = 'p_norm')
node_ml = node_ml.merge(d_conf, on = ['node_id', 'p_norm'], how = 'inner')
node_ml = node_ml.groupby('node_id').apply(lambda x: x.sample(1)).reset_index(drop=True) # there might be ties
node_ml = node_ml[['node_id', 'entry_id', 'entry_name']]

## weight of each configuration 
node_overlap = node_sum.merge(node_ml, on = 'node_id', how = 'inner')
d_all = d_merge[(d_merge['_merge'] == 'right_only') | (d_merge['_merge'] == 'both')]
d_all = d_all[['node_id', 'config_w']].drop_duplicates()
node_attr = d_all.merge(node_overlap, on = 'node_id', how = 'left', indicator = True) # should have len = n (n = 500)
node_attr.rename(columns = {'_merge': 'datastate'}, inplace = True)
node_attr.replace({'datastate': {'both': 'Yes', 'left_only': 'No'}},
                  inplace = True)
node_attr = node_attr.sort_values('node_id', ascending=True).reset_index(drop=True)

# compute hamming distance
## compute for both maximum likelihood data states (by religion)
## and for the configurations in the top n (n = 500). 
d_max = d.groupby('entry_id')['p_norm'].max().reset_index(name='p_norm')
d_max = d.merge(d_max, on = ['entry_id', 'p_norm'], how = 'inner')
datastates = d_max['p_ind'].tolist()
l_top500, l_datastates = len(p_ind), len(datastates)
bothstates = p_ind + datastates
configs = allstates[bothstates]
h_distances = compute_HammingDistance(configs) # change back
MDS_distances = h_distances[:l_top500, :l_top500]

# set-up MDS (currently baseline init values)
## MDS runs only on the top n configs 
mds = MDS(
    n_components = 2,
    n_init = 4, 
    max_iter = 300, 
    eps=1e-3, 
    random_state=seed,
    dissimilarity='precomputed',
    n_jobs=-1 
)
pos = mds.fit(MDS_distances).embedding_ # check fit, fit_transform

# couple configurations, data states and ids
comb = list(itertools.combinations(node_attr['node_id'].values, 2))

# create network
G = nx.Graph(comb)

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
### top n most plausible civilizations
n_annotate = 10
node_attr_yes = node_attr[node_attr['datastate'] == 'Yes']
df_top_n = node_attr_yes.sort_values('config_w', ascending=False).head(n_annotate)

### get position
lst_focus_nodes = df_top_n['node_id'].tolist()
labeldct = {}
positions = []
for key, val in dct_nodes.items(): 
    if key in lst_focus_nodes: 
        entry_name = val['entry_name']
        position = pos[key]
        labeldct[key] = entry_name
        positions.append((entry_name, position))

### nudge positions
nudge = [
    (0, -8), #Tsonga
    (-2, -7), #Luguru
    (-2.5, 4.5), #Thessalians (AT)
    (7, 0), #Moravian (MMN)
    (4, 3), #Islam
    (-5.05, 7.5), #Jesuits
    (-3, 8.5), #Egypt
    (-3.88, 5), #Pontifex (PMP-PC)
    (4, 5.5), #Gaudiya (GVT)
    (2, 6) #Pagan Ireland (PCR-PI)
]

transl = [
    'Tsonga',
    'Luguru',
    'Mesopotamia',
    'Cistercians',
    'Islam in Aceh',
    'LR-Selinous',
    'Ancient Egypt (OK)',
    'PMP-PC',
    'GVT',
    'PCR-PI'
]

### DOUBLE CHECK THIS, ....??
labeldct

position_nudge = {}
translation_dct = {}
for idx, val in enumerate(labeldct.values()): 
    abb = transl[idx]
    position_nudge[abb] = nudge[idx]
    translation_dct[val] = abb

translation_dct
# paths and names
perc = sum(p_vals)
out = os.path.join(outpath, f'MDS_annotated_nnodes_{n_nodes}_maxna_{maxna}_ncutoff_{n_cutoff}_perc_{perc}_seed_{seed}.pdf')

#### actually plotting ####
fig, ax = plt.subplots(facecolor = 'w', edgecolor = 'k', dpi = 300)
plt.axis('off')

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
 