'''
VMP 2022-12-15: 
Prepares key documents for the analysis of DRH data. 
This runs after 'expand_data.jl'. 
'''

import pandas as pd 
import numpy as np 

# entry_id/configuration master dataset
## load relevant data-sets 
data_expanded = pd.read_csv('/home/vmp/humanities-glass/data/analysis/data_expanded.csv')
entry_reference = pd.read_csv('/home/vmp/humanities-glass/data/analysis/entry_reference.csv')
## merge them
entry_configuration_master = data_expanded.merge(entry_reference, on = 'entry_id', how = 'inner')
## drop the entry_id_drh (can always cross-reference w. entry_reference)
entry_configuration_master = entry_configuration_master.drop(columns = ['entry_id_drh'])
## save 
entry_configuration_master.to_csv('../data/analysis/entry_configuration_master.csv', index = False)
