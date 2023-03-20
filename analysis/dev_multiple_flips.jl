# VMP 2023-03-16: run RCT simulations 
include("configuration.jl")
using .cn, Printf, Statistics, Random, Distributions, Plots, DelimitedFiles, CSV, DataFrames, StatsBase, Chain, FStrings, Base.Threads

## basic setup 
intervention_var = 2
outcome_var = 13

dir = @__DIR__
path_configuration_probabilities = replace(dir, "analysis" => "data/preprocessing/configuration_probabilities.txt")
path_configurations = replace(dir, "analysis" => "data/preprocessing/configurations.txt")
path_experiment = replace(dir, "analysis" => "data/RCT/experiment.pop.$intervention_var.$outcome_var.txt")
path_control = replace(dir, "analysis" => "data/RCT/control.pop.$intervention_var.$outcome_var.txt")

# load configurations and configuration probabilities
configuration_probabilities = readdlm(path_configuration_probabilities)
configurations = readdlm(path_configurations, Int)
configurations = cn.slicematrix(configurations)
experiment_pop = readdlm(path_experiment, Int)
control_pop = readdlm(path_control, Int)
experiment_pop .+= 1
control_pop .+= 1 

# 
intervention_idx = 5
enforce_idx = 8
n = 1
idx = collect(1:20)
perm = randperm(length(idx))
sample = idx[perm[1:2]]
sample
sample = sample[sample .!= 3]


# how does the old one work? 
flip = function(x)
    return x == 1 ? -1 : 1
end 

flip_at_index = function(index, configurations)
    new_arr = copy(configurations)
    new_arr[index] = flip(new_arr[index])
    return new_arr 
end 

# can this be used for n=1? 
flip_at_indices = function(indices, configurations)
    new_arr = copy(configurations)
    new_arr[indices] = flip.(new_arr[indices])
    return new_arr
end


proposed_flip = rand(1:20, 1)
proposed_flip
tst = flip_at_indices(proposed_flip, configurations[1])

flipped_conf = self.flip_at_index(proposed_flip)


self.new_move = function(fixed_idx = "false")
    proposed_flip = rand(1:self.len, 1)[1]
    flipped_conf = self.flip_at_index(proposed_flip)
    flipped_idx = findfirst(isequal(flipped_conf), self.states)
    if fixed_idx == proposed_flip return self end
    p_flip = self.probabilities[flipped_idx]
    move_bin = 1 - (self.p / (self.p + p_flip)) >= rand()
    # ...
    if move_bin return Configuration(flipped_idx, self.states, self.probabilities) end
    return Configuration(self.id, self.states, self.probabilities)
end 
return self 