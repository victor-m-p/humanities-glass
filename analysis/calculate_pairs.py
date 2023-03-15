'''
VMP 2023-03-08:
Strength of connections at macro-level +
Preference for individual questions at macro-level
To be compared against the parameters (Jij, hi). 
'''
import pandas as pd 
import numpy as np 
import math 
import itertools
from tqdm import tqdm 

# loads 
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype=int)
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')

# function to calculate pairs 
def calculate_pairs(configurations, configuration_probabilities, q1, q2): 
    # get probability of combinations
    q1_yes_q2_yes = configuration_probabilities[np.where((configurations[:, q1] == 1) & (configurations[:, q2] == 1))[0]].sum()
    q1_yes_q2_no = configuration_probabilities[np.where((configurations[:, q1] == 1) & (configurations[:, q2] == -1))[0]].sum()
    q1_no_q2_yes = configuration_probabilities[np.where((configurations[:, q1] == -1) & (configurations[:, q2] == 1))[0]].sum()
    q1_no_q2_no = configuration_probabilities[np.where((configurations[:, q1] == -1) & (configurations[:, q2] == -1))[0]].sum()

    # should be very close to 1 
    if not math.isclose(q1_yes_q2_yes+q1_yes_q2_no+q1_no_q2_yes+q1_no_q2_no, 1): # tolerance 1e-09
        print('sum_total is not approximately 1')
        
    # create dataframe
    d = pd.DataFrame({
        'q1': v1+1, # +1 for compatibility 
        'q2': v2+1, # +1 for compatibility 
        'q1_value': [1, 1, -1, -1],
        'q2_value': [1, -1, 1, -1],
        'probability': [q1_yes_q2_yes, q1_yes_q2_no, q1_no_q2_yes, q1_no_q2_no] 
    })
    
    return d 

# function to calculate individual questions
def calculate_mean(configurations, configuration_probabilities, question): 
    # total probability mass
    question_probability = configuration_probabilities[np.where(configurations[:, question] == 1)[0]].sum()
    d = pd.DataFrame({
        'question_id': [question+1], # +1 for compatibility
        'probability': [question_probability]
    })
    return d 

# calculate all pairs 
number_questions = len(question_reference)
combinations_list = list(itertools.combinations(range(number_questions), 2))
pairs_list = []
for v1, v2 in tqdm(combinations_list): # should be a minute  
    one_pair = calculate_pairs(configurations, configuration_probabilities, v1, v2)
    pairs_list.append(one_pair)
d_pairs = pd.concat(pairs_list, ignore_index=True)

## collapse this to the diagonal 
d_diagonal = d_pairs[d_pairs['q1_value'] == d_pairs['q2_value']]
d_diagonal = d_diagonal.groupby(['q1', 'q2']).sum().reset_index()[['q1', 'q2', 'probability']]

# calculate all individual questions
question_list = []
for question in range(number_questions): 
    one_question = calculate_mean(configurations, configuration_probabilities, question)
    question_list.append(one_question)
d_questions = pd.concat(question_list, ignore_index=True)

# save results 
d_pairs.to_csv('../data/analysis/macro_pairs.csv', index=False)
d_diagonal.to_csv('../data/analysis/macro_diagonal.csv', index=False)
d_questions.to_csv('../data/analysis/macro_questions.csv', index=False)