import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from fun import *
import seaborn as sns 
import networkx as nx 

# helper functions
def node_edge_lst(n, corr_J, means_h): 
    nodes = [node+1 for node in range(n)]
    comb = list(itertools.combinations(nodes, 2))
    d_edgelst = pd.DataFrame(comb, columns = ['n1', 'n2'])
    d_edgelst['weight'] = corr_J
    d_nodes = pd.DataFrame(nodes, columns = ['n'])
    d_nodes['size'] = means_h
    d_nodes = d_nodes.set_index('n')
    dct_nodes = d_nodes.to_dict('index')
    return d_edgelst, dct_nodes

def create_graph(d_edgelst, dct_nodes): 

    G = nx.from_pandas_edgelist(
        d_edgelst,
        'n1',
        'n2', 
        edge_attr=['weight', 'weight_abs'])

    # assign size information
    for key, val in dct_nodes.items():
        G.nodes[key]['size'] = val['size']

    # label dict
    labeldict = {}
    for i in G.nodes(): 
        labeldict[i] = i
    
    return G, labeldict

def plot_network(n_nodes, parameters, question_reference, threshold = 0.15, n_questions = 30, focus_questions = None):
    # take out parameters 
    n_J = int(n_nodes*(n_nodes-1)/2)
    J = parameters[:n_J] 
    h = parameters[n_J:]

    # get edgelist 
    d_edgelist, dct_nodes = node_edge_lst(n_nodes, J, h)
    
    if focus_questions: 
        d_edgelist = d_edgelist[(d_edgelist['n1'].isin(focus_questions)) | (d_edgelist['n2'].isin(focus_questions))]

    d_edgelist = d_edgelist.assign(weight_abs = lambda x: np.abs(x['weight']))

    # try with thresholding 
    d_edgelist_sub = d_edgelist[d_edgelist['weight_abs'] > threshold]
    G, labeldict = create_graph(d_edgelist_sub, dct_nodes)

    # different labels now 
    question_labels = question_reference.set_index('question_id')['question'].to_dict()
    labeldict = {}
    for i in G.nodes(): 
        labeldict[i] = question_labels.get(i)

    # position
    pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")

    # plot 
    fig, ax = plt.subplots(figsize = (6, 6), facecolor = 'w', dpi = 500)
    plt.axis('off')

    size_lst = list(nx.get_node_attributes(G, 'size').values())
    weight_lst = list(nx.get_edge_attributes(G, 'weight').values())
    threshold = sorted([np.abs(x) for x in weight_lst], reverse=True)[n_questions]
    weight_lst_filtered = [x if np.abs(x)>threshold else 0 for x in weight_lst]

    # vmin, vmax edges
    vmax_e = np.max(list(np.abs(weight_lst)))
    vmin_e = -vmax_e

    # vmin, vmax nodes
    vmax_n = np.max(list(np.abs(size_lst)))
    vmin_n = -vmax_n

    #size_abs = [abs(x)*3000 for x in size_lst]
    weight_abs = [abs(x)*10 for x in weight_lst_filtered]

    nx.draw_networkx_nodes(
        G, pos, 
        node_size = 400,
        node_color = size_lst, 
        edgecolors = 'black',
        linewidths = 0.5,
        cmap = cmap, vmin = vmin_n, vmax = vmax_n 
    )
    nx.draw_networkx_edges(
        G, pos,
        width = weight_abs, 
        edge_color = weight_lst, 
        alpha = 0.7, 
        edge_cmap = cmap, edge_vmin = vmin_e, edge_vmax = vmax_e)
    nx.draw_networkx_labels(G, pos, font_size = 8, labels = labeldict)
    plt.show(); 
       
# setup 
seed = 1
cmap = plt.cm.coolwarm
cutoff_n = 35

# question labels 
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
question_reference = question_reference[['question_id', 'question']]

# load files 
n_nodes, n_nan, n_rows, n_entries = 20, 5, 455, 407
basepath = f'../data/mdl_experiments/matrix_questions_{n_nodes}_maxna_{n_nan}_nrows_{n_rows}_entries_{n_entries}'
params_hidden = np.loadtxt(f'{basepath}.txt.mpf_HIDDEN_params.dat')
params_removed = np.loadtxt(f'{basepath}.txt.mpf_REMOVED_params.dat')
params_original = np.loadtxt(f'{basepath}.txt.mpf_params.dat') 
params_added = np.loadtxt(f'{basepath}.txt.mpf_ADDED_params.dat')

# plot the original constraint network
plot_network(n_nodes = 20,
             parameters = params_original,
             question_reference = question_reference,
             threshold = 0.15,
             n_questions = 35)

plot_network(n_nodes = 20,
             parameters = params_original,
             question_reference = question_reference, 
             threshold = 0,
             n_questions = 15,
             focus_questions = [1])

# plot the constraint network with removed question
# but the node is negative, so it really boosts reincarnation in this world 
question_reference_removed = question_reference[question_reference['question_id'] != 1]
question_reference_removed['question_id'] = [x-1 for x in question_reference_removed['question_id']]
plot_network(n_nodes = 19,
             parameters = params_removed,
             question_reference = question_reference_removed,
             threshold = 0.15,
             n_questions = 30)

# infers a hidden question constraining reincarnation in this world 
question_reference_hidden = question_reference.copy()
question_reference_hidden['question'] = question_reference_hidden.apply(lambda row: 'hidden' if row['question_id'] == 1 else row['question'], axis = 1)

plot_network(n_nodes = 20,
             parameters = params_hidden,
             question_reference = question_reference_hidden,
             threshold = 0.15,
             n_questions = 35)

plot_network(n_nodes = 20,
             parameters = params_hidden,
             question_reference = question_reference_hidden,
             threshold = 0,
             n_questions = 15,
             focus_questions=[1])

# infers a hidden node boosting reincarnation in this world
# this node is positive so it actually does boost 
question_reference_added = pd.concat([question_reference, pd.DataFrame({'question_id': [0], 'question': ['added']})])
question_reference_added = question_reference_added.sort_values('question_id')
question_reference_added['question_id'] = [x+1 for x in question_reference_added['question_id']]
question_reference_added = question_reference_added.reset_index(drop=True)

plot_network(n_nodes = 21,
             parameters = params_added,
             question_reference = question_reference_added,
             threshold = 0.15,
             n_questions = 35)

plot_network(n_nodes = 21,
             parameters = params_added,
             question_reference = question_reference_added, 
             threshold = 0,
             n_questions = 15,
             focus_questions = [1])

##### can we make the setup comparable? ####

##### what are the actual parameter differences? #####
parameters = params_hidden
threshold = 0.15
n_J = int(n_nodes*(n_nodes-1)/2)
J = parameters[:n_J] 
h = parameters[n_J:]

# get edgelist 
d_edgelist, dct_nodes = node_edge_lst(n_nodes, J, h)
d_edgelist = d_edgelist[(d_edgelist['n1'].isin(focus_questions)) | (d_edgelist['n2'].isin(focus_questions))]
d_edgelist = d_edgelist.assign(weight_abs = lambda x: np.abs(x['weight']))

# try with thresholding 
d_edgelist_sub = d_edgelist[d_edgelist['weight_abs'] > threshold]
G, labeldict = create_graph(d_edgelist_sub, dct_nodes)

# different labels now 
question_labels = question_reference.set_index('question_id')['question'].to_dict()
labeldict = {}
for i in G.nodes(): 
    labeldict[i] = question_labels.get(i)

# position
pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")

size_lst = list(nx.get_node_attributes(G, 'size').values())
weight_lst = list(nx.get_edge_attributes(G, 'weight').values())

weight_lst


threshold = sorted([np.abs(x) for x in weight_lst], reverse=True)[30]
weight_lst_filtered = [x if np.abs(x)>threshold else 0 for x in weight_lst]

# vmin, vmax edges
vmax_e = np.max(list(np.abs(weight_lst)))
vmin_e = -vmax_e

# vmin, vmax nodes
vmax_n = np.max(list(np.abs(size_lst)))
vmin_n = -vmax_n

#size_abs = [abs(x)*3000 for x in size_lst]
weight_abs = [abs(x)*10 for x in weight_lst_filtered]

# plot 
fig, ax = plt.subplots(figsize = (6, 6), facecolor = 'w', dpi = 500)
plt.axis('off')


nx.draw_networkx_nodes(
    G, pos, 
    node_size = 400,
    node_color = size_lst, 
    edgecolors = 'black',
    linewidths = 0.5,
    cmap = cmap, vmin = vmin_n, vmax = vmax_n 
)
nx.draw_networkx_edges(
    G, pos,
    width = weight_abs, 
    edge_color = weight_lst, 
    alpha = 0.7, 
    edge_cmap = cmap, edge_vmin = vmin_e, edge_vmax = vmax_e)
nx.draw_networkx_labels(G, pos, font_size = 8, labels = labeldict)
plt.show(); 