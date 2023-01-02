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

conf = Configuration(idx, configurations, configuration_probabilities)

# get information out 
conf.to_string()
conf.configuration
conf.p 

# implement push_forward

# implement bit_flip(): 
flip = lambda x: -1 if x == 1 else 1 

# so, we can flip a bit, but ... 
# requires us to make a copy ... 
def flip_index(array, index): 
    new_arr = np.copy(array)
    new_arr[index] = flip(new_arr[index])
    return new_arr 

def hamming_neighbors(array): 
    arr_lst = [] 
    for num, _ in enumerate(array): 
        new_arr = flip_index(array, num)
        arr_lst.append(new_arr)
    return arr_lst 

flips = hamming_neighbors(test)

config_ids = [np.where((configurations == i).all(1))[0][0] for i in flips]
config_probs = configuration_probabilities[config_ids]

def array_sum_to_one(array): 
    array = array / array.min()
    array = array / array.sum()
    return array 

normalized_array = array_sum_to_one(config_probs)
array_indices = np.arange(20)
x = np.random.choice(array_indices, size=1, p=normalized_array)
x = x[0]

# test whether this works (it works)
l = []
for i in range(1000): 
    x = np.random.choice(array_indices, size = 1, p = normalized_array)
    x = x[0]
    l.append(x)

d = pd.DataFrame(l, columns = ['id'])
normalized_array.max()
np.where(normalized_array > 0.1)
d.groupby('id').size().reset_index(name = 'size').sort_values('size', ascending = False)





# implement hamming neighbors
def get_n_neighbors(n_neighbors, idx_focal, config_allstates, prob_allstates):
    config_focal = config_allstates[idx_focal]
    prob_focal = prob_allstates[idx_focal]
    lst_neighbors = []
    for idx_neighbor, config_neighbor in enumerate(config_allstates): 
        h_dist = np.count_nonzero(config_focal!=config_neighbor)
        if h_dist <= n_neighbors and idx_focal != idx_neighbor: 
            prob_neighbor = prob_allstates[idx_neighbor]
            lst_neighbors.append((idx_focal, prob_focal, idx_neighbor, prob_neighbor, h_dist ))
    df_neighbor = pd.DataFrame(
        lst_neighbors, 
        columns = ['idx_neighbor', 'prob_neighbor', 'hamming']
    )
    return df_neighbor



### old stuff ###
y = "".join([str(x) if x == 1 else str(0) for x in configx])
y

class Civilization: 
    def __init__(self, entry)