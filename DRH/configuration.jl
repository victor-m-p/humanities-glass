# install these packages into environment (e.g. enter Pkg REPL and use 'add [PKG]').
using Printf, Statistics, Distributions, DelimitedFiles, CSV, DataFrames, IterTools

# little function here 
function slicematrix(A::AbstractMatrix)
    return [A[i, :] for i in 1:size(A,1)]
end

# class
mutable struct Configuration 
    # all the input variables we take 
    id # int something perhaps 
    configurations # some kind of array
    configuration_probabilities # some kind of array

    # all the attributes we want 
    configuration
    p 

    # the functions as well (which is a bit quirky)
    get_probability 
    get_configuration
    get_transition 

    # the main body 
    function Configuration(id, configurations, configuration_probabilities)
        self = new()

        self.id = id 
        # functions to run on init 
        ## get probability function
        self.get_probability = function(configuration_probabilities)
            return configuration_probabilities[self.id]
        end 

        ## get configuration function 
        self.get_configuration = function(configurations)
            return configurations[self.id]
        end
        
        self.configuration = self.get_configuration(configurations) 
        self.p = self.get_probability(probabilities)

        # hamming distance 
        #self.hamming_neighbors = function()

        # transition probabilities 
        self.get_transition = function(configurations, configuration_probabilities, enforce_move = False)
            hamming_array = self.hamming_neighbors()
            if not enforce_move  
                # not 100% sure whether copy() is needed
                hamming_array = push!(copy(hamming_array), self.configuration)
            end 

        end 

        this.configuration = self.get_configuration(configurations) 
        this.p = self.get_probability(probabilities)

        return this 
    end 
end 

# load shit 
configuration_probabilities = readdlm("/home/vmp/humanities-glass/data/analysis/configuration_probabilities.txt")
configurations = readdlm("/home/vmp/humanities-glass/data/analysis/configurations.txt", Int)
## I need an array, rather than a matrix 
configurations = slicematrix(configurations)

# NB: configurations should be 
ex = Configuration(1, configurations, configuration_probabilities)
ex.p
ex.configuration # why did this change? (global scope stuff?)

# works 
function flip(x)
    return x == 1 ? -1 : 1
end 

# works 
function flip_index(x, index)
    new_arr = copy(x)
    new_arr[index] = flip(new_arr[index])
    return new_arr 
end 

# array sum to one 
## consider dropping because it is so simple 
function normalize(array)
    return array./sum(array, dims = 1)
end 

## hamming (works, I think). 
function hamming_neighbors(arr)
    hamming_list = []
    for (num, _) in enumerate(arr)
        tmp_arr = flip_index(arr, num)
        append!(hamming_list, [tmp_arr])
    end 
    return Vector(hamming_list)
end 




