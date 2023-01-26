import pandas as pd 
import networkx as nx 
import numpy as np 
import configuration as cn 
import re
import matplotlib.pyplot as plt 
import os 
from tqdm import tqdm 

# preprocessing 
from fun import bin_states 
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
n_nodes = 20
configurations = bin_states(n_nodes) 

text = 'Jeg ølsker én 你好'
from unidecode import unidecode
unidecode(text).strip()



files = os.listdir('../data/COGSCI23/attractors')
for file in tqdm(files): 
    config_id = int(re.match(r't0.5_max5000_idx(\d+).csv', file)[1])
    d = pd.read_csv(f'../data/COGSCI23/attractors/{file}')
    d = d[['config_from', 'config_to', 'probability']].drop_duplicates()

    # find strength of self-loops 
    config_from = d['config_from'].unique().tolist()
    config_to = d['config_to'].unique().tolist()
    config_total = list(set(config_from + config_to))

    # can take a litle bit 
    # could move to julia 
    remain_probability = []
    for idx in config_total: 
        ConfObj = cn.Configuration(idx, 
                                configurations,
                                configuration_probabilities)
        p_move = ConfObj.p_move(configurations,
                                configuration_probabilities)
        p_stay = 1 - p_move 
        remain_probability.append((idx, p_stay))

    remain_probability = pd.DataFrame(remain_probability, columns = ['config_id', 'P(remain)'])

    # observed maximum-likelihood state & label 
    pd.set_option('display.max_colwidth', None) 
    entry_maxlikelihood = pd.read_csv('../data/analysis/entry_maxlikelihood.csv')
    entry_maxlikelihood = entry_maxlikelihood[['config_id', 'entry_name']]
    entry_maxlikelihood = entry_maxlikelihood.groupby('config_id')['entry_name'].apply(lambda x: "\n".join(x)).reset_index(name = 'entry_list')
    entry_maxlikelihood['entry_list'] = [re.sub(r"(\(.*\))|(\[.*\])", "", x) for x in entry_maxlikelihood['entry_list']]
    entry_maxlikelihood['entry_list'] = [unidecode(text).strip() for text in entry_maxlikelihood['entry_list']]
    node_attributes = remain_probability.merge(entry_maxlikelihood, on = 'config_id', how = 'left').fillna("")
    node_attributes['node_color'] = ['tab:blue' if x else 'tab:orange' for x in node_attributes['entry_list']]
    node_attributes['node_color'] = ['tab:red' if x == config_id else y for x, y in zip(node_attributes['config_id'], node_attributes['node_color'])]

    source = node_attributes[node_attributes['node_color'] == 'tab:red']['entry_list'].values[0]
    source = re.split('\n', source)[0]

    # create network 
    d = d.rename(columns = {'probability': 'weight'})
    G = nx.from_pandas_edgelist(d, source = 'config_from',
                                target = 'config_to', edge_attr = 'weight',
                                create_using = nx.DiGraph)

    # add node information
    labels = {}
    for _, row in node_attributes.iterrows(): 
        config_id = row['config_id']
        G.nodes[config_id]['p_remain'] = row['P(remain)']
        G.nodes[config_id]['node_color'] = row['node_color']
        labels[config_id] = row['entry_list']

    # extract information
    node_size = list(nx.get_node_attributes(G, 'p_remain').values())
    node_color = list(nx.get_node_attributes(G, 'node_color').values())
    edge_size = list(nx.get_edge_attributes(G, 'weight').values())

    # pos 
    pos = nx.kamada_kawai_layout(G, weight = 'weight')

    # plot 
    fig, ax = plt.subplots(dpi = 300)
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size = [x*25 for x in node_size], node_color = node_color)
    nx.draw_networkx_edges(G, pos, width = edge_size, edge_color = 'tab:grey')
    nx.draw_networkx_labels(G, pos, labels = labels, font_size = 6)
    plt.suptitle(f'{source}', size = 15)
    plt.savefig(f'../fig/attractors/{source}_{config_id}.pdf')