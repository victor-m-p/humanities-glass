import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return dendrogram(linkage_matrix, **kwargs)


iris = load_iris()

X = iris.data

# get our data in here
n_nodes = 20 
from fun import bin_states, top_n_idx
allstates = bin_states(n_nodes) 
config_prob = np.loadtxt('../data/analysis/configuration_probabilities.txt')
n_top_states = 150
top_config_info = top_n_idx(n_top_states, config_prob, 'config_id', 'config_prob') 
top_config_info['node_id'] = top_config_info.index
configuration_ids = top_config_info['config_id'].tolist()
top_configurations = allstates[configuration_ids]

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, 
                                n_clusters=None)

model = model.fit(top_configurations)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# Q: how do we extract things 
# *: there is some color threshold argument
fig, ax = plt.subplots(figsize = (8, 15), dpi = 300)
dendrogram_dict = plot_dendrogram(model, 
                                  orientation = 'left',
                                  get_leaves = True)
plt.savefig('../fig/dendrogram.pdf')

# extract information
leaves = dendrogram_dict.get('leaves')
leaves_color = dendrogram_dict.get('leaves_color_list')
leaf_dataframe = pd.DataFrame(
    {'node_id': leaves,
     'node_cluster': leaves_color}
)

# save information
leaf_dataframe.to_csv('../data/analysis/dendrogram_clusters.csv', index = False)