# Not used in the COGSCI23 submission
# VMP 2023-02-08: refactored and re-run.
include("configuration.jl")
using .cn, Printf, Statistics, Distributions, DelimitedFiles, CSV, DataFrames, IterTools, StatsBase, Chain, FStrings, Base.Threads

# manage paths 
dir = @__DIR__
path_configuration_probabilities = replace(dir, "analysis" => "data/preprocessing/configuration_probabilities.txt")
path_configurations = replace(dir, "analysis" => "data/preprocessing/configurations.txt")
#path_entry_config = replace(dir, "analysis" => "data/preprocessing/entry_maxlikelihood.csv")
path_question_reference = replace(dir, "analysis" => "data/preprocessing/question_reference.csv")
path_network_information = replace(dir, "analysis" => "data/analysis/top_configurations_network.csv")

# load configurations and configuration probabilities
configuration_probabilities = readdlm(path_configuration_probabilities)
configurations = readdlm(path_configurations, Int)
configurations = cn.slicematrix(configurations)

# load all maximum likelihood configurations 
## might not be necessary depending on what we want to look at ##
#entry_maxlikelihood = DataFrame(CSV.File(path_entry_config))
network_information = DataFrame(CSV.File(path_network_information))

# load question reference
question_reference = DataFrame(CSV.File(path_question_reference))

# find the ones with not punishing & monitoring Gods
config_id = network_information[!, "config_id"]
config_id = config_id .+ 1 # python indexing 

# variables
q1 = 12
q2 = 13

# find the observed (maxlik) configurations 
# that have q1 and q2 as -1 
observed_configurations = configurations[config_id]
observed_config_id = [findfirst(isequal(i), configurations) for i in observed_configurations]
observed_mat = mapreduce(permutedims, vcat, observed_configurations)
starting_configurations = findall((observed_mat[:, q1] .== -1) .& (observed_mat[:, q2] .== -1))
starting_config_id = observed_config_id[starting_configurations]
starting_config_id = 361985 # Messalians 

ConfStart = cn.Configuration(starting_config_id, configurations, configuration_probabilities)
idx_neighbors, p_neighbors = ConfStart.id_and_prob_of_neighbors()

n_simulation = 200 # first simulation is self so this gives n = 100
n_timestep = 200 # first timestep is self so this gives n = 10
global sample_list = [] 
global conf_list = []
global n_neighbors = 1

@time begin 
for unique_config in idx_neighbors
    for sim_number in 1:n_simulation
        x = findfirst(isequal(unique_config), [x for (x, y) in conf_list]) # is this what we want?
        if x isa Number 
            ConfObj = conf_list[x][2] # return the corresponding class 
        else 
            ConfObj = cn.Configuration(unique_config, configurations, configuration_probabilities)
        end 
        id = ConfObj.id 
        for time_step in 1:n_timestep
            push!(sample_list, [sim_number, time_step, starting_config_id, id])
            if id ∉ [x for (x, y) in conf_list]
                push!(conf_list, [id, ConfObj]) 
            end 
            ConfObj = ConfObj.move(n_neighbors, conf_list)
            id = ConfObj.id 
        end 
    end 
end  
end 

println("saving file")
df = DataFrame(
simulation = [a for (a, _, _, _) in sample_list],
timestep = [b for (_, b, _, _) in sample_list],
starting_config = [c-1 for (_, _, c, _) in sample_list],
config_id = [d-1 for (_, _, _, d) in sample_list] # -1 for python indexing
)

outpath = replace(dir, "analysis" => f"data/sim/rct_messalians.csv")
CSV.write(outpath, df)