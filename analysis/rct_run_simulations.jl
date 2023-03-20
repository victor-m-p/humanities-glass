# VMP 2023-03-16: run RCT simulations 
include("configuration.jl")
using .cn, Printf, Statistics, Random, Distributions, Plots, DelimitedFiles, CSV, DataFrames, StatsBase, Chain, FStrings, Base.Threads

# function for running simulations 
function run_mixed_simulation(
    n,
    intervention_var,
    outcome_var, 
    sample_pop,
    n_time_enforced,
    n_time_free,
    outname,
    configurations=configurations,
    configuration_probabilities=configuration_probabilities)

    simulation = [] 
    for idx in sample_pop
        configObj = cn.Configuration(idx, configurations, configuration_probabilities)
        enforce_intervention=intervention_var
        for t in 1:n_time_enforced
            configObj = configObj.new_move(enforce_intervention, n)
            config = configObj.configuration
            intervention, outcome = config[intervention_var], config[outcome_var]
            push!(simulation, [t, idx, configObj.id, enforce_intervention, intervention, outcome])
        end 
        enforce_intervention=false
        for t in n_time_enforced+1:n_time_enforced+n_time_free
            configObj = configObj.new_move(enforce_intervention)
            config = configObj.configuration
            intervention, outcome = config[intervention_var], config[outcome_var]
            push!(simulation, [t, idx, configObj.id, enforce_intervention, intervention, outcome])
        end
    end 

    matrix = hcat(simulation...)'
    d = DataFrame(matrix, [:timestep, :config_id, :config_id_new, :enforce_intervention, :intervention, :outcome])
    d.outcome = ifelse.(d.outcome .== -1, 0, d.outcome)
    d.intervention = ifelse.(d.intervention .== -1, 0, d.intervention)
    outpath = replace(dir, "analysis" => "data/RCT2/$outname.csv")
    CSV.write(outpath, d)
end 

## basic setup 
intervention_var = 5
outcome_var = 10

# manage paths 
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

## run the straight forward stuff 
sample_ids = ["experiment", "control"]
samples = [experiment_pop, control_pop]
n_fixed_timesteps = [0, 30, 100]
n_free_timesteps = [100, 70, 0]

for (sample_id, sample) in zip(sample_ids, samples)
    for (fixed_steps, free_steps) in zip(n_fixed_timesteps, n_free_timesteps)
        n = rand(1:2, 1)[1]
        run_mixed_simulation(
            n,
            intervention_var,
            outcome_var,
            sample, 
            fixed_steps,
            free_steps,
            "pct50.$sample_id.$fixed_steps.$free_steps.$intervention_var.$outcome_var"
        )
    end 
end 