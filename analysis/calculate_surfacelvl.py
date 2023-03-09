'''
Calculate correlations and means for all questions in the survey.
To be compared with macro-level probability weight and inferred parameters.
'''

import numpy as np 
import pandas as pd 
import itertools 

# read data in appropriate format and only consider full observations
d = pd.read_csv('../data/reference/direct_reference_questions_20_maxna_5_nrows_455_entries_407.csv')
d = d[d['weight'] > 0.99]
d = d.drop(columns=['entry_id', 'weight'])

# correlations 
## recode columns of d 
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
question_dictionary = question_reference.set_index('question_id_drh')['question_id'].to_dict()
question_dictionary = {str(k): v for k, v in question_dictionary.items()}
d = d.rename(columns=question_dictionary)

## wrangle data 
param_corr = d.corr(method='pearson')
param_corr['q1'] = param_corr.index
param_corr_melt = pd.melt(param_corr, id_vars = 'q1', value_vars = question_dictionary.values(), 
                          value_name = 'correlation', var_name = 'q2')
param_corr_melt = param_corr_melt[param_corr_melt['q1'] < param_corr_melt['q2']]

# means 
param_mean = d.mean().reset_index(name = 'mean')
param_mean = param_mean.rename(columns = {'index': 'question_id'})

# save data
param_corr_melt.to_csv('../data/analysis/raw_correlations.csv', index = False)
param_mean.to_csv('../data/analysis/raw_means.csv', index = False)