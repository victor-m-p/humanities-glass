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
        self.len = len(self.configuration)
        self.enforce_move = np.nan
        
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
    def normalize(array): 
        array = array / array.min()
        array = array / array.sum()
        return array 

    # flip probabilities (including or excluding self)
    # make sure that this checks as well whether it is already computed 
    def get_transition(self, configurations, 
                                 configuration_probabilities, 
                                 enforce_move = False):
        
        # if already computed with same settings then just return
        if self.enforce_move == enforce_move: 
            return self.config_ids, self.config_probs 
        
        # check whether it is already computed 
        hamming_array = self.hamming_neighbors()
        
        # if enforce move we do not add self 
        if not enforce_move: 
            hamming_array = np.concatenate([hamming_array, [self.configuration]], axis = 0)
             
        # get configuration ids, and configuration probabilities
        self.config_ids = [np.where((configurations == i).all(1))[0][0] for i in hamming_array]
        self.config_probs = configuration_probabilities[self.config_ids]
        self.enforce_move = enforce_move 
        
        # return 
        return self.config_ids, self.config_probs         
    
    # flip probability to specific other ..?
    
    # hamming neighbors
    # NB: need to solve the problem of not recomputing
    def hamming_neighbors(self): # for now only immediate neighbors
        hamming_lst = [] 
        for num, _ in enumerate(self.configuration): 
            tmp_arr = self.flip_index(num)
            hamming_lst.append(tmp_arr)
        hamming_array = np.array(hamming_lst)
        return hamming_array 

    # would be nice to do so that it can take another class 
    # other assumed to be a class as well 
    def hamming_distance(self, other): 
        x = self.configuration 
        y = other.configuration
        array_overlap = (x == y)
        h_distance = len(x) - sum(array_overlap)
        return h_distance 
  
    # used by overlap, diverge
    def answer_comparison(self, other, question_reference):  
        answers = pd.DataFrame([(x, y) for x, y in zip(self.configuration, other.configuration)], 
                                columns = [self.id, other.id])
        answers = pd.concat([question_reference, answers], axis = 1)
        return answers
    
    # overlap in answers between two configuration instances 
    def overlap(self, other, question_reference): 
        answers = self.answer_comparison(other, question_reference)
        answers_overlap = answers[answers[self.id] == answers[other.id]]
        return answers_overlap
    
    # difference in answers between two configuration instances 
    def diverge(self, other, question_reference):  
        answers = self.answer_comparison(other, question_reference)
        answers_nonoverlap = answers[answers[self.id] != answers[other.id]]
        return answers_nonoverlap 
  
    def neighbor_probabilities(self, configurations, configuration_probabilities, 
                               question_reference, enforce_move = False, top_n = False):
        # if enforce move it is simple 
        if enforce_move: 
            config_ids, config_probs = self.get_transition(configurations, configuration_probabilities, enforce_move = True)
            d = pd.DataFrame([(config_id, config_prob) for config_id, config_prob in zip(config_ids, config_probs)],
                            columns = ['config_id', 'config_prob'])
            d = pd.concat([d, question_reference], axis = 1)
            d[self.id] = self.configuration
        # else it is a bit more complicated 
        else: 
            config_ids, config_probs = self.get_transition(configurations, configuration_probabilities, enforce_move = True)
            d = pd.DataFrame([(config_id, config_prob) for config_id, config_prob in zip(config_ids, config_probs)],
                        columns = ['config_id', 'config_prob'])
            self_columns = question_reference.columns
            self_row = pd.DataFrame([['remain', 'remain', 'remain', 'remain']], columns = self_columns)
            question_referencex = pd.concat([question_reference, self_row])
            question_referencex = question_referencex.reset_index(drop = True)
            d = pd.concat([d, question_referencex], axis = 1)
            d[self.id] = list(self.configuration) + [0] 
        # common for both 
        d['transition_prob'] = d['config_prob']/d['config_prob'].sum()
        d = d.sort_values('transition_prob', ascending = False)
        # if we are only interested in the most probable n neighboring probabilities 
        if top_n: 
            d = d.head(top_n)
        return d 
    
    def probability_remain(self, configurations, configuration_probabilities, n = 0): 
        _, config_prob_neighbors = self.get_transition(configurations, configuration_probabilities, enforce_move = True)
        config_prob_neighbors = np.sum(np.sort(config_prob_neighbors)[:self.len-n])
        prob_remain = self.p/(self.p+config_prob_neighbors) # 
        return prob_remain
  
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
        config_ids, config_probs = self.get_transition(configurations, 
                                                                    configuration_probabilities,
                                                                    enforce_move)
        
        # either sample probabilistically
        if probabilistic:
            transition_normalized = self.normalize(config_probs) 
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
#entry_configuration_master = pd.read_csv('../data/analysis/entry_configuration_master.csv')
#configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
#question_reference = pd.read_csv('../data/analysis/question_reference.csv')

# generate all states
#n_nodes = 20
#from fun import bin_states 
#configurations = bin_states(n_nodes) 

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