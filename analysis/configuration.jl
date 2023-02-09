#Convenience class for configurations. 
#VMP 2023-02-05: clean up, and converge with the corresponding class. 

# install these packages into environment (e.g. enter Pkg REPL and use 'add [PKG]').
module cn
using Printf, Statistics, Distributions, DelimitedFiles, CSV, DataFrames, IterTools, StatsBase, Chain, FStrings

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
    id_neighbor::Vector{Int64}
    p_neighbor::Vector{Float64}
    transition::Bool
    len::Int64
    prob_targets
    prob_move
    targets 
    states::Vector{Vector{Int64}}
    probabilities::Matrix{Float64}

    # the functions as well (which is a bit quirky)
    get_probability 
    get_configuration
    match_row 
    flip 
    flip_at_index 
    flip_at_indices
    normalize 
    hamming_neighbors 
    id_and_prob_of_neighbors
    neighbor_id 
    p_move 
    move 

    # the main body 
    function Configuration(id, configurations, configuration_probabilities)
        self = new()
        self.id = id 
        self.states = configurations 
        self.probabilities = configuration_probabilities
        self.transition = false
        self.prob_move = false 

        # functions to run on init 
        # get probability function
        self.get_probability = function(configuration_probabilities)
            return configuration_probabilities[self.id]
        end 

        # get configuration function 
        self.get_configuration = function(configurations)
            return configurations[self.id]
        end

        # should be static?
        self.match_row = function(x, configurations) 
            return findfirst(isequal(x), configurations)
        end 
        
        self.configuration = self.get_configuration(configurations) 
        self.p = self.get_probability(configuration_probabilities)
        self.len = length(self.configuration)

        self.flip = function(x)
            return x == 1 ? -1 : 1
        end 

        self.flip_at_index = function(index)
            new_arr = copy(self.configuration)
            new_arr[index] = self.flip(new_arr[index])
            return new_arr 
        end 

        # refactored 
        self.flip_at_indices = function(indices)
            new_arr = copy(self.configuration)
            new_arr[indices] = self.flip.(new_arr[indices])
            return new_arr
        end

        self.neighbor_id = function(indices) 
            return [findfirst(isequal(self.flip_at_index(i)), self.states) for i in indices]
        end 

        # should be static?
        self.normalize = function(arr)
            return arr./sum(arr, dims = 1)
        end 

        # refactored 
        self.hamming_neighbors = function()
            return map(self.flip_at_index, 1:length(self.configuration))
        end

        # refactored
        self.id_and_prob_of_neighbors = function()
            if self.transition
                return self.id_neighbor, self.p_neighbor
            end

            hamming_array = self.hamming_neighbors()
            self.id_neighbor = map(x -> self.match_row(x, self.states), hamming_array)
            self.p_neighbor = self.probabilities[self.id_neighbor]
            self.transition = true

            return self.id_neighbor, self.p_neighbor
        end

        # refactored
        self.p_move = function(summary = true)
            _, p_neighbor = self.id_and_prob_of_neighbors()
            self.prob_move = 1 .- (self.p ./ (self.p .+ p_neighbor))
            if !summary 
                return self.prob_move
            else 
                return mean(self.prob_move) 
            end 
        end 

        # refactored 
        self.move = function(n, conf_list = false)
            # initialize self.prob_move if not yet computed
            if self.prob_move == false self.p_move(false) end 
            
            # determine targets to flip
            self.targets = rand(1:self.len, n)
            self.prob_targets = self.prob_move[self.targets]
            move_bin = self.prob_targets .>= rand(n)

            # return self if no flip occurs
            if !any(move_bin) return self end 
            
            # for a single flip
            if n == 1
                new_id = self.id_neighbor[self.targets][1]
                if conf_list != false
                    x = findfirst(isequal(new_id), [x for (x, y) in conf_list])
                    if x isa Number return conf_list[x][2] end 
                end
                return Configuration(new_id, self.states, self.probabilities)
            end

            # for multiple flips
            feature_changes = [x for (x, y) in zip(self.targets, move_bin) if y]
            new_configuration = self.flip_at_indices(feature_changes)
            new_id = self.match_row(new_configuration, self.states)
            if conf_list != false 
                x = findfirst(isequal(new_id), [x for (x, y) in conf_list])
                if x isa Number return conf_list[x][2] end 
            end
            return Configuration(new_id, self.states, self.probabilities)
        end
        return self 
    end 
end
end 