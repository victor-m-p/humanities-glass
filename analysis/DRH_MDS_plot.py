import numpy as np 
import pandas as pd 
from sim_fun import bin_states, compute_HammingDistance
from sklearn.manifold import MDS
import itertools
import networkx as nx 
import matplotlib.pyplot as plt
import os 

# setup 
n_nodes, maxna = 20, 10
seed = 2
n_cutoff = 500
outpath = '../fig'

# read files 
p_file = '../data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt'
mat_file = '../data/clean/matrix_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.txt'
d_main = '../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
sref = '../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
nref = '../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
p = np.loadtxt(p_file)
d_main = pd.read_csv(d_main)
sref = pd.read_csv(sref)
nref = pd.read_csv(nref)
#datastates_weighted = np.loadtxt(mat_file)
#datastates = np.delete(datastates_weighted, n_nodes, 1)
#datastates_uniq = np.unique(datastates, axis = 0) # here we loose correspondence

# get all state configurations 
allstates = bin_states(n_nodes) # takes a minute (do not attempt with n_nodes > 20)

#### for julia test ####
outname = '../data/analysis/allstates_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt'
np.savetxt('../data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt', p)

# subset states above threshold
val_cutoff = np.sort(p)[::-1][n_cutoff]
p_ind = [i for i,v in enumerate(p) if v > val_cutoff]
p_vals = p[p > val_cutoff]
substates = allstates[p_ind]
perc = round(np.sum(p_vals)*100,2) 

# compute hamming distance
distances = compute_HammingDistance(substates) 

# set this up when we have a plot we like
mds = MDS(
    n_components = 2,
    n_init = 4, # number initializations of SMACOF alg. 
    max_iter = 300, # set up after testing
    eps=1e-3, # set up after testing
    random_state=seed,
    dissimilarity='precomputed',
    n_jobs=-1 # check this if it is slow 
)
pos = mds.fit(distances).embedding_ # check fit, fit_transform

# couple configurations, data states and ids
substates_id = [(num, ele) for num, ele in enumerate(substates)]
substates_id = pd.DataFrame(substates_id, columns = ['index', 'config']) # this is good
comb = list(itertools.combinations(substates_id['index'].values, 2))

# overlap between data states and configs while preserving linking
d_cols = d_main.columns
d_Q_uniq = d_main.drop_duplicates(list(d_cols[1:-1]))
d_Q_uniq = d_Q_uniq.rename(columns = {'s': 'entry_id'})
d_Q_uniq = d_Q_uniq.merge(sref, on = 'entry_id', how = 'inner')
df_substates = pd.DataFrame(substates, columns = list(d_cols[1:-1]))
df_substates['node_index'] = df_substates.index
d_Q_uniq = d_Q_uniq.convert_dtypes() # preserve dtypes after nan merge
df_substates = df_substates.convert_dtypes() # preserve dtypes after nan merge
df_overlap = df_substates.merge(d_Q_uniq, on = list(d_cols[1:-1]), how = 'left', indicator = True)

# gather node attributes
node_attr = df_overlap[['node_index', 'entry_id', 'entry_name', '_merge']]
node_attr = node_attr.rename(columns = {'_merge': 'datastate'})
node_attr.replace({'datastate': {'both': 'Yes', 'left_only': 'No'}}, inplace = True)
node_attr['p_ind'] = p_ind
node_attr['size'] = p_vals

# create network
G = nx.Graph(comb)

dct_nodes = node_attr.to_dict('index')
for key, val in dct_nodes.items():
    G.nodes[key]['index'] = val['node_index']
    G.nodes[key]['size'] = val['size']
    G.nodes[key]['datastate'] = val['datastate']
    G.nodes[key]['i_ind'] = val['p_ind']
    G.nodes[key]['entry_id'] = val['entry_id']
    G.nodes[key]['entry_name'] = val['entry_name']

# create node list for subplot
ids = list(nx.get_node_attributes(G, 'index').values())
vals = list(nx.get_node_attributes(G, 'datastate').values())
nodelst = [id if val == "Yes" else "" for id, val in zip(ids, vals)]
nodelst = list(filter(None, nodelst))

# prepare plot
node_scaling = 2000
size_lst_raw = list(nx.get_node_attributes(G, 'size').values())
size_lst = [x*node_scaling for x in size_lst_raw]

# make raw plot #
out = os.path.join(outpath, f'MDS_nnodes_{n_nodes}_maxna_{maxna}_ncutoff_{n_cutoff}_perc_{perc}_seed_{seed}.pdf')
fig, axis = plt.subplots(facecolor = 'w', edgecolor = 'k', dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size = size_lst)
nx.draw_networkx_nodes(G, pos, nodelist = nodelst, node_size = 15,
                       node_color = 'black', node_shape = 'x')
plt.savefig(out)

#### annotated plot ####
# top n most plausible civilizations
node_attr_data = node_attr[node_attr['datastate'] == 'Yes']
node_attr_data.sort_values('size', ascending=False).head(5)
df_top_n = node_attr_data.sort_values('size', ascending=False).head(5)

# a couple of interesting outliers 
other_focus_civs = pd.DataFrame({
    'node_index': [31, # interesting outlier
                   82, # interesting outlier
                   #332, # outlier to the left
                   ]})
df_focus_qual = node_attr_data.merge(other_focus_civs, on = 'node_index', how = 'inner')

node_attr_data[node_attr_data['Christian'] == True]

# get entry and position
df_focus_nodes = pd.concat([df_top_n, df_focus_qual])
lst_focus_nodes = df_focus_nodes['node_index'].tolist()
labeldct = {}
positions = []
for key, val in dct_nodes.items(): 
    if key in lst_focus_nodes: 
        entry_name = val['entry_name']
        position = pos[key]
        labeldct[key] = entry_name
        positions.append((entry_name, position))

# nudge position
position_nudge = {
    'Spiritualism': (0, -5.5),
    'Secular Buddhists': (1, -4),
    'Tsonga': (-1, -7),
    '12th-13th c Cistercians': (5, 0),
    'Local Religion at Selinous': (2, 4),
    'Cham Bani': (-4, 6),
    'Bön (Bon)': (-0.5, 5)
}

# plot 
out = os.path.join(outpath, f'MDS_annotated_nnodes_{n_nodes}_maxna_{maxna}_ncutoff_{n_cutoff}_perc_{perc}_seed_{seed}.pdf')
fig, axis = plt.subplots(facecolor = 'w', edgecolor = 'k', dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size = size_lst, node_color = 'tab:blue')
nx.draw_networkx_nodes(G, pos, nodelist = nodelst, node_size = 15,
                       node_color = 'tab:orange', node_shape = 'x')
# annotate 
for entry_name, position in positions: 
    pos_x, pos_y = position
    x_nudge, y_nudge = position_nudge.get(entry_name)
    axis.annotate(entry_name, xy=[pos_x, pos_y], 
                  xytext=[pos_x+x_nudge, pos_y+y_nudge],
                  arrowprops = dict(arrowstyle="->",
                                    connectionstyle="arc3"))
# save
plt.savefig(out)

#### reference plot #####
d_not_data = node_attr[node_attr['datastate'] == 'No'].sort_values('size', ascending = False)
nodelst_not_data = d_not_data.head(5)['node_index'].tolist()

## find biggest nodes that are data states
d_data = node_attr[node_attr['datastate'] == 'Yes'].sort_values('size', ascending = False)
nodelst_data = d_data.head(5)['node_index'].tolist()

## get labels for nodes of interest
all_interest = nodelst_not_data + nodelst
labeldict = {}
for key, val in dct_nodes.items():
    if key in all_interest:
        labeldict[key] = val['node_index'] # should change
    else: 
        pass

## reference plot ##
out = os.path.join(outpath, f'MDS_reference_nnodes_{n_nodes}_maxna_{maxna}_ncutoff_{n_cutoff}_perc_{perc}_seed_{seed}.pdf')
fig, axis = plt.subplots(facecolor = 'w', edgecolor = 'k', dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size = size_lst, node_color = 'tab:blue')
nx.draw_networkx_nodes(G, pos, nodelist = nodelst_not_data, node_size = 15,
                       node_color = 'tab:orange', node_shape = 'x')
nx.draw_networkx_nodes(G, pos, nodelist = nodelst_data, node_size = 15,
                       node_color = 'tab:red', node_shape = 'x')
nx.draw_networkx_labels(
    G, pos, 
    labels = labeldict, 
    font_size = 5,
    font_color='black')
plt.savefig(out)

## some important nodes
d_important_points = pd.DataFrame({
    'index': [31, 82, 439, 486, 438, 274, 285, 59],
    'desc': [
        'datastate_outlier_center_bottom', #31
        'datastate_outlier_left_bottom', #82
        'datastate_top5_centroid_top', #439
        'datastate_top5_centroid_mid', #486
        'datastate_top5_centroid_bot', #438
        'datastate_top5_left_mid', #274
        'config_top5_left_mid_l', #285
        'config_top5_left_mid_r' #59
        ]})

## data states not in top 500 configurations 
## hamming distance 
distances.shape # 500, 500
substates # the 500 configs in the plot 
substates

# get all state configurations 
allstates = bin_states(n_nodes) # takes a minute (do not attempt with n_nodes > 20)

# subset states above threshold
val_cutoff = np.sort(p)[::-1][n_cutoff]
p_ind = [i for i,v in enumerate(p) if v > val_cutoff]
p_vals = p[p > val_cutoff]
substates = allstates[p_ind]
perc = round(np.sum(p_vals)*100,2) 

#### test number of rows without any nan ####
d_cols = d_main.columns
d_A = np.array(d_main[list(d_cols[1:-1])])
d_A[np.all(d_A, axis = 1)].shape
