# install these packages into environment (e.g. enter Pkg REPL and use 'add [PKG]').

module cn # begin module
using Printf, Statistics, Distributions, DelimitedFiles, CSV, DataFrames, IterTools, StatsBase, Chain

# little function here 
function slicematrix(A::AbstractMatrix)
    return [A[i, :] for i in 1:size(A,1)]
end

# class
mutable struct Configuration 
    # all the input variables we take 
    id::Int # int something perhaps 
    configurations::Vector{Vector{Int64}} # some kind of array
    configuration_probabilities::Matrix{Float64} # some kind of array

    # all the attributes that can be defined  
    configuration::Vector{Int64}
    p::Float64
    config_ids::Vector{Int64}
    config_probs::Vector{Float64}
    enforce_move 

    # the functions as well (which is a bit quirky)
    get_probability 
    get_configuration
    match_row 
    flip 
    flip_index 
    normalize 
    hamming_neighbors 
    get_transition 
    push_forward 

    # the main body 
    function Configuration(id, configurations, configuration_probabilities)
        self = new()

        self.id = id 
        self.enforce_move = 100 # not sure how to give it e.g. a NAN

        # functions to run on init 
        ## get probability function
        self.get_probability = function(configuration_probabilities)
            return configuration_probabilities[self.id]
        end 

        ## get configuration function 
        self.get_configuration = function(configurations)
            return configurations[self.id]
        end

        ## should be static 
        self.match_row = function(x, configurations) 
            return findfirst(isequal(x), configurations)
        end 
        
        self.configuration = self.get_configuration(configurations) 
        self.p = self.get_probability(configuration_probabilities)

        # works 
        self.flip = function(x)
            return x == 1 ? -1 : 1
        end 

        # works 
        self.flip_index = function(index)
            new_arr = copy(self.configuration)
            new_arr[index] = self.flip(new_arr[index])
            return new_arr 
        end 

        # normalize array 
        self.normalize = function(arr)
            return arr./sum(arr, dims = 1)
        end 

        # hamming distance 
        self.hamming_neighbors = function()
            hamming_list = []
            for (num, _) in enumerate(self.configuration)
                tmp_arr = self.flip_index(num)
                append!(hamming_list, [tmp_arr])
            end 
            return Vector(hamming_list)
        end 

        # transition probabilities 
        self.get_transition = function(configurations, configuration_probabilities, enforce_move = false)
            
            # need to implement that it should not recompute 
            # would be nice to speed test to verify 
            if self.enforce_move == enforce_move
                return self.config_ids, self.config_probs 
            end 

            hamming_array = self.hamming_neighbors()
            if !enforce_move  
                # not 100% sure whether copy() is needed
                hamming_array = push!(copy(hamming_array), self.configuration)
            end 

            # get configuration ids, and configuration probabilities 
            self.config_ids = [self.match_row(x, configurations) for x in hamming_array] # works 
            self.config_probs = configuration_probabilities[self.config_ids]
            self.enforce_move = enforce_move 

            # return 
            return self.config_ids, self.config_probs 
        end 

        # push forward 
        self.push_forward = function(configurations, configuration_probabilities, probabilistic = true, enforce_move = false, conf_list = false)
            
            # with or without enforcing move 
            config_ids, config_probs = self.get_transition(configurations, configuration_probabilities, enforce_move)

            # sample probabilistically 
            if probabilistic 
                transition_normalized = self.normalize(config_probs)
                sample_id = sample(config_ids, Weights(transition_normalized))

                # saving computation (hopefully) (NB: get rid of nesting, and implement for non-self-sample)
                x = findfirst(isequal(sample_id), [x for (x, y) in conf_list])
                if x isa Number 
                    return conf_list[x][2] # return the corresponding class 
                end 

                # ...
                return Configuration(sample_id, configurations, configuration_probabilities)

            # sample deterministically 
            else 
                sample_id = findmax(config_probs)[2]
                sample_id = config_ids[sample_id]
                return Configuration(sample_id, configurations, configuration_probabilities)
            end 

        end 

        return self 
    end 
end 

end # end module 