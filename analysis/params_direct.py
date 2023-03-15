'''
VMP 2023-02-28: check fits for simulated data with hidden and removed nodes 
'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import os 
import re 
import itertools 
import seaborn as sns 

def parse_file(params, type, idx = None): 
    
    # basic decomposition
    n_params = len(params)
    n_nodes = int(0.5 * (np.sqrt(8*n_params+1)-1))
    n_J = int(n_nodes*(n_nodes-1)/2)
    Jij = params[:n_J]
    hi = params[n_J:]

    # handle different types 
    if type == 'removed': 
        n_questions = len(hi)+1
        combination_list = [x+1 for x in range(n_questions) if x != idx]
    elif type == 'added': 
        n_questions = len(hi)
        combination_list = [x for x in range(n_questions)]
    else: 
        n_questions = len(hi)
        combination_list = [x+1 for x in range(n_questions)]

    # generate Jij indices 
    Jij_idx = list(itertools.combinations(combination_list, 2))

    # create dataframes 
    d_Jij = pd.DataFrame({
        'type': [type for _ in range(len(Jij))], 
        #'id': [idx for _ in range(len(Jij))],
        'Jij': Jij_idx, 
        'Ji': [i[0] for i in Jij_idx],
        'Jj': [i[1] for i in Jij_idx],
        'coupling': Jij
    })
    d_hi = pd.DataFrame({
        'type': [type for _ in range(len(hi))], 
        #'id': [idx for _ in range(len(hi))],
        'hi': combination_list,
        'field': hi 
    })

    return d_Jij, d_hi 

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

Jij_original, hi_original = parse_file(params_original, 'original')
Jij_removed, hi_removed = parse_file(params_removed, 'removed', 0)
Jij_hidden, hi_hidden = parse_file(params_hidden, 'hidden')
Jij_added, hi_added = parse_file(params_added, 'added')

# put these together 
Jij_master = pd.concat([Jij_original, Jij_removed, Jij_hidden, Jij_added])
hi_master = pd.concat([hi_original, hi_removed, hi_hidden, hi_added])

###### Jij overview ######
Jij_shared = Jij_master[Jij_master['type'] == 'removed'][['Jij']]
Jij_shared = Jij_master.merge(Jij_shared, on = 'Jij', how = 'inner')

# sort by original
Jij_sort = Jij_shared[Jij_shared['type'] == 'original'][['Jij', 'coupling']]
Jij_sort = Jij_sort.sort_values(by = 'coupling').reset_index(drop = True)
Jij_sort['idx'] = Jij_sort.index
Jij_sort = Jij_sort[['Jij', 'idx']]
Jij_shared_sort = Jij_shared.merge(Jij_sort, on = 'Jij', how = 'inner')

## plot this (generally close)
def sns_scatter(data, param, lims, s = 15): 
    sns.scatterplot(x='groundtruth', y=param,
                    hue='type', data=data,
                    s = s)
    plt.plot(lims, lims, 'k-')

sns.scatterplot(data=Jij_shared_sort, x="coupling", y='idx', 
                s = 15, hue = 'type')

###### hi overview ######
hi_shared = hi_master[hi_master['type'] == 'removed'][['hi']]
hi_shared = hi_master.merge(hi_shared, on = 'hi', how = 'inner')

# sort by original
hi_sort = hi_shared[hi_shared['type'] == 'original'][['hi', 'field']]
hi_sort = hi_sort.sort_values(by = 'field').reset_index(drop = True)
hi_sort['idx'] = hi_sort.index
hi_sort = hi_sort[['hi', 'idx']]
hi_shared_sort = hi_shared.merge(hi_sort, on = 'hi', how = 'inner')

sns.scatterplot(data=hi_shared_sort, x="field", y='idx',
                s = 15, hue = 'type')

# look at some of the biggest deviations
Jij_original = Jij_shared_sort[Jij_shared_sort['type'] == 'original']
Jij_original = Jij_original.rename(columns = {'coupling': 'original'})
Jij_original = Jij_original[['Jij', 'original']]

Jij_changes = Jij_shared_sort.merge(Jij_original, on = 'Jij', how = 'inner')
Jij_changes['change'] = Jij_changes['coupling'] - Jij_changes['original']
Jij_changes['abs_change'] = np.abs(Jij_changes['change'])

# first average 
Jij_changes.groupby('type')['abs_change'].mean() # added changes the least; hidden a bit further, but very close

# then specifics
## add question reference
question_reference_ji = question_reference.rename(columns = {'question_id': 'Ji', 'question': 'question_i'})
question_reference_jj = question_reference.rename(columns = {'question_id': 'Jj', 'question': 'question_j'})

Jij_changes = Jij_changes.merge(question_reference_ji, on = 'Ji', how = 'inner')
Jij_changes = Jij_changes.merge(question_reference_jj, on = 'Jj', how = 'inner')

Jij_removed = Jij_changes[Jij_changes['type'] == 'removed']
Jij_removed.sort_values('abs_change', ascending = False).head(10)
# monumental architecture (makes sense because)