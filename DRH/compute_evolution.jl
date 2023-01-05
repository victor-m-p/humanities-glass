include("configuration.jl")
using .cn, Printf, Statistics, Distributions, DelimitedFiles, CSV, DataFrames, IterTools, StatsBase, Chain, FStrings

# load shit 
configuration_probabilities = readdlm("/home/vmp/humanities-glass/data/analysis/configuration_probabilities.txt")
configurations = readdlm("/home/vmp/humanities-glass/data/analysis/configurations.txt", Int)
## I need an array, rather than a matrix 
configurations = cn.slicematrix(configurations)

# load all maximum likelihood configurations 
entry_config_filename = "/home/vmp/humanities-glass/data/analysis/entry_maxlikelihood.csv"
entry_maxlikelihood = DataFrame(CSV.File(entry_config_filename))
config_ids = @chain entry_maxlikelihood begin _.config_id end
unique_configs = unique(config_ids) # think right, but double check 
unique_configs = unique_configs .+ 1 # because of 0-indexing in python 

# setup 
n_simulation = 100
n_timestep = 100
batch_size = 10
sample_list = [] 
conf_list = []
total_configs = length(unique_configs)
@time begin 
for (num, unique_config) in enumerate(unique_configs)
    println("$num / $total_configs")
    lx = length(sample_list)
    println("length: $lx")
    for sim_number in 1:n_simulation
        x = findfirst(isequal(unique_config), [x for (x, y) in conf_list]) # is this what we want?
        if x isa Number 
            ConfObj = conf_list[x][2] # return the corresponding class 
        else 
            ConfObj = cn.Configuration(unique_config, configurations, configuration_probabilities)
        end 
        id = ConfObj.id 
        for time_step in 1:n_timestep
            push!(sample_list, [sim_number, time_step, id])
            if id ∉ [x for (x, y) in conf_list]
                push!(conf_list, [id, ConfObj]) 
            end 
            ConfObj = ConfObj.push_forward(configurations, configuration_probabilities, true, false, conf_list)
            id = ConfObj.id 
        end 
    end 
    # save as we go for larger ones 
    # memory becomes pretty wild 
    if num % batch_size == 0
        println("saving file")
        d = DataFrame(
        simulation = [x for (x, y, z) in sample_list],
        timestep = [y for (x, y, z) in sample_list],
        config_id = [z for (x, y, z) in sample_list]
        )
        CSV.write(f"/home/vmp/humanities-glass/data/COGSCI23/evo/s_{n_simulation}_t_{n_timestep}_n_{num}.csv", d)
        global sample_list = []
    end 
end 
end 

x = []
length(x)