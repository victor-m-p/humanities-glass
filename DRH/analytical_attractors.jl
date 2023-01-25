# COGSCI23
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

# find id of neighbors
function get_neighbor_idx(configuration, indices, configurations)
    neighbor_idx_above_threshold = []
    for i in indices
        flip = configuration.flip_index(i)
        flipid = findfirst(isequal(flip), configurations)
        append!(neighbor_idx_above_threshold, flipid)
    end 
    return neighbor_idx_above_threshold 
end 

# setup
max_timestep = 100
threshold = 0.5 
#max_neighbors = 3
total_configs = length(unique_configs)

# big loop 
global n_config = 0
for original_idx in unique_configs 
    global n_config += 1
    println("$n_config / $total_configs")
    df = DataFrame(
        timestep = Int64[],
        config_from = Int64[],
        config_to = Int64[],
        probability = Float64[]
    )
    focal_idx_list = original_idx 

    for t in 1:max_timestep+1
        # get all of our moves 
        neighbor_idx_total = [] 
        for focal_idx in focal_idx_list 
            ConfObj = cn.Configuration(focal_idx, configurations, configuration_probabilities)
            p_move = ConfObj.p_move(configurations, configuration_probabilities, false)
            indices = findall(>(threshold), p_move)
            values = p_move[indices]
            if length(indices) != 0
                #if length(indices) > max_neighbors
                #    top_n = sortperm(values)[1:max_neighbors]
                #    indices = indices[top_n]
                #    values = values[top_n]
                #end 
                neighbor_idx_list = get_neighbor_idx(ConfObj, indices, configurations)
                append!(neighbor_idx_total, neighbor_idx_list)
                for pair in zip(neighbor_idx_list, values)
                    neighbor_idx, values = pair 
                    push!(df, [t-1, focal_idx-1, neighbor_idx-1, values])
                end 
            end 
        if length(neighbor_idx_total) == 0 
            break 
        else 
            focal_idx_list = neighbor_idx_total 
        end 
    end 
    end 
    original_idx = original_idx .- 1
    CSV.write(f"/home/vmp/humanities-glass/data/COGSCI23/attractors/t{threshold}_max{max_timestep}_idx{original_idx}.csv", df)
end 