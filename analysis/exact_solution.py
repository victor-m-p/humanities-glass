'''
VMP 2023-03-20
Trying to get an exact solution.
Also, this code is fucking crazy
and should change the way we structure
the class in any case. 
'''

import numpy as np 
import pandas as pd 
import configuration as cn 

# this is breaking news 
p = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
configurations = np.loadtxt('../data/preprocessing/configurations.txt')

def glauber_n1(p, i, neighbors, num_variables): 
    '''
    p: array of probabilities
    i: index of current configuration
    neighbors: indices of neighboring configurations
    '''
    p_neighbors = p[neighbors]
    p_self = p[i]
    p_move = p_neighbors / (p_self + p_neighbors)
    p_move = p_move / num_variables
    p_stay = np.sum(p_move)
    return p_move, p_stay

### ChatGPT code ###
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def generate_transition_matrix(num_variables, p):
    num_states = 2**num_variables
    transition_matrix = lil_matrix((num_states, num_states))

    for i in range(num_states):
        # create neighbor list 
        neighbors = []
        for j in range(num_variables):
            # Calculate the index of the neighboring state by flipping the j-th bit
            neighbor = i ^ (1 << j)
            neighbors.append(neighbor)
        
        # get probability for move 
        p_move, p_stay = glauber_n1(p, i, neighbors, num_variables)
        transition_matrix[i, i] = p_stay
        # insert probabilities
        for j in range(num_variables): 
            transition_matrix[i, neighbors[j]] = p_move[j]

    return transition_matrix.tocsr()

# Initial probability distribution
num_variables = 20
initial_probs = p  # Should have a length of 2**20

# Generate the sparse transition matrix
transition_matrix = generate_transition_matrix(num_variables, p)

# Evolve the system for a given number of steps
# Okay, this crashes my local computer
# Try on the server tomorrow. 
num_steps = 2
current_probs = initial_probs

for _ in range(num_steps):
    current_probs = current_probs.dot(transition_matrix)

current_probs

'''
Replace your_probability_here with the actual transition probabilities for your system and your_probabilities_here with your initial probability distribution.

This code generates a sparse transition matrix and evolves the system for a given number of steps. Note that we use the lil_matrix format for constructing the sparse matrix, as it allows efficient row-based modifications. After constructing the matrix, we convert it to the csr_matrix format, which is more efficient for matrix multiplication.
'''