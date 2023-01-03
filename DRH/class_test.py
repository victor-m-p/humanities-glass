import pandas as pd 
import numpy as np
from tqdm import tqdm 

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

    # flip a bit (static)
    @staticmethod
    def flip(x): 
        return -1 if x == 1 else 1 
    
    # flip a specific index in array 
    def flip_index(self, index): 
        new_arr = np.copy(self.configuration)
        new_arr[index] = self.flip(new_arr[index]) # not sure whether this should be "flip" or "self.flip"
        return new_arr 

    # sum array to one (static)
    @staticmethod
    def array_sum_to_one(array): 
        array = array / array.min()
        array = array / array.sum()
        return array 

    # flip probabilities (including or excluding self)
    # make sure that this checks as well whether it is already computed 
    def transition_probabilities(self, configurations, 
                                 configuration_probabilities, enforce_move = False):
        # check whether it is already computed 
        hamming_array = self.hamming_neighbors()
        
        # if enforce move we do not add self 
        if not enforce_move: 
            hamming_array = np.concatenate([hamming_array, [self.configuration]], axis = 0)
             
        # get configuration ids, and configuration probabilities
        config_ids = [np.where((configurations == i).all(1))[0][0] for i in hamming_array]
        config_probs = configuration_probabilities[config_ids]
        
        # return 
        return config_ids, config_probs         
    
    # flip probability to specific other ..?
    
    # hamming neighbors
    # NB: need to solve the problem of not recomputing
    def hamming_neighbors(self): # for now only immediate neighbors
        #if self.hamming_array: 
        #    return hamming_array
        #else: 
        hamming_lst = [] 
        for num, _ in enumerate(self.configuration): 
            tmp_arr = self.flip_index(num)
            hamming_lst.append(tmp_arr)
        hamming_array = np.array(hamming_lst)
        #self.hamming_array = hamming_array
        return hamming_array 

    # would be nice to do so that it can take another class 
    # other assumed to be a class as well 
    def hamming_distance(self, other): 
        x = self.configuration 
        y = other.configuration
        array_overlap = (x == y)
        h_distance = len(x) - sum(array_overlap)
        return h_distance 
  
    # naive path between two configuration
    def naive_path(other): 
        pass 
    
    # move the configuration one step 
    # could be 
    # (1) probabilistic
    # (2) deterministic
    # (x) include/exclude probability to stay
    def push_forward(self, configurations, configuration_probabilities,
                     probabilistic = True, enforce_move = False):
        
        # with or without enforcing move 
        config_ids, config_probs = self.transition_probabilities(configurations, 
                                                                    configuration_probabilities,
                                                                    enforce_move)
        
        # either sample probabilistically
        if probabilistic:
            transition_normalized = self.array_sum_to_one(config_probs) 
            transition_ids = np.arange(len(transition_normalized))
            sample = np.random.choice(transition_ids, size=1, p=transition_normalized)
            sample = sample[0]
            
        # or deteministically take the maximum 
        else: 
            sample = np.argmax(config_probs)
            
        sample_id = config_ids[sample]
        return Configuration(sample_id, configurations, configuration_probabilities)
        
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

## experiment 
track_list = []
idx = 769927 # Roman Imperial Cult 
config_orig = Configuration(idx, configurations, configuration_probabilities)
config = Configuration(idx, configurations, configuration_probabilities)
h_distance = 0
probability = config.get_probability(configuration_probabilities)
for i in tqdm(range(1000)): 
    track_list.append((idx, h_distance, probability))
    config = config.push_forward(configurations,
                                 configuration_probabilities)
    idx = config.id 
    h_distance = config_orig.hamming_distance(config)
    probability = config.get_probability(configuration_probabilities)

# to pandas
d = pd.DataFrame(track_list, columns = ['config_id', 'hamming', 'prob'])
d = d[['config_id', 'hamming']]
d.to_csv('../data/push_forward/random_n_1000_config_769927.csv', index = False)

## this we can actually plot in interesting ways ...
## i.e. we can do it as in the earlier DeDeo work. 
## or if we run a lot of iterations (needs to be more efficient)
## then we can just summarize the avg. time in neighborhood
## (defined as any hamming distance we want, but e.g. 0, 1, or 2). 
## ALSO: (annotate which other known religions it hits). 
## ALSO: (Simon had the idea with floodplains, so ...)
## we could do this for the top 150 communities and gather
## some kind of information on how stable they are 
## e.g. % stay on same, % stay close (e.g. within 2H) % stay in comm. 

# want a function which tells me the difference 
# between two different configurations 
# i.e. I actually want to know which 
# questions they disagree about. 


# options: 
## (1) move probabilistically, possible to stay
## (2) move probabilistically, enforce move 
## (3) move to specific neighbor 
## (4) hamming distance to other idx 
## (5) probability to move to other idx neighbor (enforce move)
## (6) probability to move to other idx neighbor (possible to stay)

class Civilization: 
    def __init__(self, entry)