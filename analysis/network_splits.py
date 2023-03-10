'''
VMP 2023-03-09: 
Try to color the network by splits in attributes,
e.g. high Gods vs. not high Gods. 
'''

# imports 
import pandas as pd 
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import itertools 

# question reference
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
number_questions = len(question_reference)
combinations_list = list(itertools.combinations(range(number_questions), 2))

for q1, q2 in tqdm(combinations_list): # should be a minute  

    q1_label = question_reference[question_reference['question_id'] == q1+1]['question_short'].values[0]
    q2_label = question_reference[question_reference['question_id'] == q2+1]['question_short'].values[0]

    # read data
    network_information = pd.read_csv('../data/analysis/top_configurations_network.csv')
    hamming_information = pd.read_csv('../data/analysis/top_configurations_hamming.csv')

    # find the actual configurations
    configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)
    top_configs = configurations[network_information['config_id'].values, :]

    q1_yes_q2_yes = list(np.where((top_configs[:, q1] == 1) & (top_configs[:, q2] == 1))[0])
    q1_yes_q2_no = list(np.where((top_configs[:, q1] == 1) & (top_configs[:, q2] == -1))[0])
    q1_no_q2_yes = list(np.where((top_configs[:, q1] == -1) & (top_configs[:, q2] == 1))[0])
    q1_no_q2_no = list(np.where((top_configs[:, q1] == -1) & (top_configs[:, q2] == -1))[0])

    d_combinations = pd.DataFrame({
        'index': q1_yes_q2_yes + q1_yes_q2_no + q1_no_q2_yes + q1_no_q2_no,
        'combination': ['yy' for _, _ in enumerate(q1_yes_q2_yes)] + ['yn' for _, _ in enumerate(q1_yes_q2_no)] + ['ny' for _, _ in enumerate(q1_no_q2_yes)] + ['nn' for _, _ in enumerate(q1_no_q2_no)],
    })

    d_combinations = d_combinations.sort_values('index').reset_index(drop = True)
    network_information = pd.merge(network_information, d_combinations, left_index=True, right_index=True)

    # assign color for the combination
    color_coding = {
        'yy': '#800000', # green
        'yn': '#3cb44b', # yellow
        'ny': '#f032e6', # red
        'nn': '#a9a9a9'} # blue

    network_information['color'] = network_information['combination'].map(color_coding)

    # create network
    G = nx.from_pandas_edgelist(hamming_information,
                                'node_x',
                                'node_y',
                                'hamming')

    # extract position
    pos = {}
    for idx, row in network_information.iterrows():
        node_id = row['node_id']
        pos_x = row['pos_x']
        pos_y = row['pos_y']
        pos[node_id] = (pos_x, pos_y)

    ## add node information to the graph 
    network_information_dict = network_information.to_dict('index')
    for idx, val in network_information_dict.items(): 
        node_id = val['node_id'] # should also be idx but cautious
        for attr in val: 
            G.nodes[node_id][attr] = val[attr]

    # process network information
    from fun import * 
    G = edge_strength(G, 'config_prob') # would be nice to get rid of this. 
    edgelist_sorted, edgeweight_sorted = sort_edge_attributes(G, 'pmass_mult', 'hamming', 34000)
    nodelist_sorted, nodesize_sorted = sort_node_attributes(G, 'config_prob', 'config_prob')
    _, community_sorted = sort_node_attributes(G, 'config_prob', 'color') 
    node_scalar = 13000

    # plot without labels 
    fig, ax = plt.subplots(figsize = (8, 8), dpi = 500)
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, 
                            nodelist = nodelist_sorted,
                            node_size = [x*node_scalar for x in nodesize_sorted], 
                            node_color = community_sorted,
                            linewidths = 0.5, edgecolors = 'black')
    nx.draw_networkx_edges(G, pos, alpha = 0.7,
                        width = edgeweight_sorted,
                        edgelist = edgelist_sorted,
                        edge_color = 'tab:grey')

    custom_lines = [Line2D([0], [0], marker='o', label='Scatter', color = color_coding['yy'], markersize=10),
                    Line2D([0], [0], marker='o', label='Scatter', color = color_coding['yn'], markersize=10),
                    Line2D([0], [0], marker='o', label='Scatter', color = color_coding['ny'], markersize=10),
                    Line2D([0], [0], marker='o', label='Scatter', color = color_coding['nn'], markersize=10)]

    handles = [f'{q1_label} = yes, {q2_label} = yes',
               f'{q1_label} = yes, {q2_label} = no',
               f'{q1_label} = no, {q2_label} = yes',
               f'{q1_label} = no, {q2_label} = no']

    ax.legend(custom_lines, handles, bbox_to_anchor=(0.89, 0.05), fontsize = 8)
    plt.savefig(f'../fig/network_splits_png/{q1_label}_{q2_label}.png', bbox_inches = 'tight')
    plt.clf()