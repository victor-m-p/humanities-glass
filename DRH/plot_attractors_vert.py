'''
VMP 2022-01-30: this is the one we actually use. 
'''

import networkx as nx             # graph manipulation
import numpy as np                # numerical utilities
import matplotlib.pyplot as plt   # plotting
import pandas as pd               # format results
import re
import os 
np.random.seed(0)              # set seed for reproducibility
import configuration as cn 
from fun import * 
from tqdm import tqdm 
from unidecode import unidecode

# https://www.color-hex.com/color-palette/68785
clrs = [
    '#0097c3',
    '#9be3f9',
    '#ffef55',
    '#f24f26',
] 

entry_maxlikelihood = pd.read_csv('../data/analysis/entry_maxlikelihood.csv')
entry_maxlikelihood = entry_maxlikelihood[['config_id', 'entry_name']]
entry_maxlikelihood = entry_maxlikelihood.groupby('config_id').sample(n=1, random_state=1)
entry_maxlikelihood['entry_name'] = [re.sub(r"(\(.*\))|(\[.*\])", "", x) for x in entry_maxlikelihood['entry_name']]
entry_maxlikelihood['entry_name'] = [re.sub(r"\/", " ", x) for x in entry_maxlikelihood['entry_name']]
entry_maxlikelihood['entry_name'] = [unidecode(text).strip() for text in entry_maxlikelihood['entry_name']]

files = os.listdir('../data/COGSCI23/attractors')

for file in tqdm(files): 
    config_orig = int(re.match(r't0.5_max5000_idx(\d+).csv', file)[1])

    d = pd.read_csv(f'../data/COGSCI23/attractors/{file}')
    d = d[['config_from', 'config_to', 'probability']].drop_duplicates()
    
    if not len(d) == 0: 
        # get probability for each state for vertical axis 
        config_from = d['config_from'].unique().tolist()
        config_to = d['config_to'].unique().tolist()
        config_uniq = list(set(config_from + config_to))

        configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
        p = configuration_probabilities[config_uniq]
        config_probs = pd.DataFrame({
            'config_id': config_uniq,
            'config_prob': p
        })

        config_probs['log_config_prob'] = [np.log(x) for x in config_probs['config_prob']]

        ## node size by hamming? ## 
        configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
        n_nodes = 20
        configurations = bin_states(n_nodes) 
        ConfOrig = cn.Configuration(config_orig,
                                    configurations,
                                    configuration_probabilities)
        hamming_distances = []
        for i in config_uniq: 
            ConfOther = cn.Configuration(i,
                                        configurations,
                                        configuration_probabilities)
            distance = ConfOrig.hamming_distance(ConfOther)
            hamming_distances.append((i, distance))

        hamming_distances = pd.DataFrame(hamming_distances,
                                        columns = ['config_id', 'hamming'])

        # fix node colors
        attractors = list(set(config_to) - set(config_from))
        unique_config_df = pd.DataFrame({'config_id': config_uniq})
        node_attributes = unique_config_df.merge(entry_maxlikelihood, on = 'config_id', how = 'left').fillna("")
        node_attributes['node_color'] = [clrs[0] if x else clrs[1] for x in node_attributes['entry_name']]
        node_attributes['node_color'] = [clrs[2] if x == config_orig else y for x, y in zip(node_attributes['config_id'], node_attributes['node_color'])]
        node_attributes['node_color'] = [clrs[3] if x in attractors else y for x, y in zip(node_attributes['config_id'], node_attributes['node_color'])]

        # for logging
        source = node_attributes[node_attributes['config_id'] == config_orig]['entry_name'].tolist()[0]
        
        # add data
        naive_path = pd.read_csv(f'../data/COGSCI23/max_attractor/idx{config_orig}.csv')
        naive_path = naive_path[['config_from', 'config_to']]
        naive_path['edge_color'] = 'k'
        d = d.merge(naive_path, on = ['config_from', 'config_to'], how = 'left').fillna('tab:grey')
        d['edge_width'] = [x*3 if y == 'k' else x*1.5 for x, y in zip(d['probability'], d['edge_color'])]

        G = nx.from_pandas_edgelist(d, 
                                    source = 'config_from',
                                    target = 'config_to',
                                    edge_attr = ['edge_color', 'edge_width'],
                                    create_using = nx.DiGraph)

        # now we start adding stuff 
        node_attr = config_probs.merge(node_attributes, on = 'config_id', how = 'inner')
        node_attr = node_attr.merge(hamming_distances, on = 'config_id', how = 'inner')
        for _, row in node_attr.iterrows(): 
            config_id = int(row['config_id'])
            G.nodes[config_id]['log_config_prob'] = row['log_config_prob']
            G.nodes[config_id]['node_color'] = row['node_color']
            G.nodes[config_id]['hamming'] = row['hamming']

        # get node attributes out 
        pos = {}
        nodelist = []
        node_color = []
        node_size = []
        for node, attr in G.nodes(data = True): 
            pos[node] = np.array([np.random.normal(0, 0.1), attr['log_config_prob']]) 
            nodelist.append(node)
            node_color.append(attr['node_color'])
            node_size.append(attr['hamming'])

        # get edge attributes out 
        edge_dict = {}
        for config_from, config_to, attr in G.edges(data = True): 
            col = attr['edge_color']
            size = attr['edge_width']
            edge_dict[(config_from, config_to)] = [size, col]

        edge_dict_sorted = {k: v for k, v in sorted(edge_dict.items(), key=lambda item: item[1])}
        edgelist_sorted = list(edge_dict_sorted.keys())

        edge_color = []
        edge_width = []
        for x in list(edge_dict_sorted.values()):
            edge_width.append(x[0])
            edge_color.append(x[1])
        # visualize the graph
        plt.figure(dpi = 300)
        plt.axis('off')
        nx.draw_networkx_nodes(G, 
                            pos = pos, 
                            nodelist = nodelist,
                            node_color = node_color,
                            node_size = [(x+1)*20 for x in node_size],
                            edgecolors = 'k',
                            linewidths = 1)
        nx.draw_networkx_edges(G, 
                            pos = pos,
                            edgelist = edgelist_sorted,
                            edge_color = edge_color,
                            width = edge_width)
        plt.savefig(f'../fig/attractor_plots/{source}_{config_orig}.pdf')
        plt.close()
