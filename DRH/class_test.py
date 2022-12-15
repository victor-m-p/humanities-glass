import pandas as pd 
import numpy as np 

class Configuration: 
    def __init__(self, 
                 id, 
                 states, 
                 probabilities):
        self.id = id
        self.configuration = self.get_configuration(states)
        self.p = self.get_probability(probabilities)
        
    # consider: is entry something we NEED to have
    # or is entry something that we can add when needed?
   
    # as long dataframe with columns (configuration_id, question_id, question_name, question_name_drh) 
    def to_long():
        pass
    
    # as a wide dataframe where questions are columns
    def to_wide(): 
        pass 
    
    # as a matrix 
    def to_matrix(): 
        pass 
    
    # as a string 
    def to_string(self):
        return "".join([str(x) if x == 1 else str(0) for x in self.configuration])
    
    # depends on whether we require this property
    # otherwise we might include to possibility
    # to add this attribute 
    def get_probability(self, probabilities): 
        probability = probabilities[self.id]
        return probability

    # again depends on whether we require this 
    # otherwise we might include the possibility
    # to add this attribute
    def get_configuration(self, configurations): 
        configuration = configurations[self.id]
        return configuration

    # flip probabilities (including or excluding self)
    def transition_probabiliies():
        pass 
    
    # flip probability to specific other ..?
    
    # hamming neighbors
    def hamming_neighbors(n):
        pass 

    # naive path between two configuration
    def naive_path(other): 
        pass 
    
    # move the configuration one step 
    # could be 
    # (1) probabilistic
    # (2) deterministic
    # (x) include/exclude probability to stay
    def push_forward():
        pass 
    
    # instantiate civilization class
    def to_civilization(x): 
        pass 

# load documents
entry_configuration_master = pd.read_csv('../data/analysis/entry_configuration_master.csv')
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')

# generate all states
n_nodes = 20
from fun import bin_states 
configurations = bin_states(n_nodes) 

# check some functionality for a single configuration
idx = 769975
config = configurations[idx]
prob = configuration_probabilities[idx]
configuration_probabilities

conf = Configuration(idx, configurations, configuration_probabilities)
conf.to_string()

configx = conf.configuration
y = "".join([str(x) if x == 1 else str(0) for x in configx])
y

class Civilization: 
    def __init__(self, entry)