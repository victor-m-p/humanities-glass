using Printf
using Statistics 
using Distributions
using DelimitedFiles
include("fun.jl")

# setup
beta = 1
m = 1000
n = 30

# init
par, h_offset = sample_params(n, beta) # big list 
obs = rand([-1, 1], m, n) # obs 
samp = copy(obs)
ij = get_index(n)

# start MCMC 
# MCMC(pos_x, iter, n)
function MCMC(pos_x, iter, n, samp)
    for _ in [1:1:iter;]
        pos_y = rand([1:1:n;])
        running = 0
        for j in [1:1:n;]
            if j != pos_y 
                running -= (samp[pos_x, pos_y] - flip_bit(samp[pos_x, pos_y])) * samp[pos_x, j] * par[ij[pos_y, j]]
            end 
        end
        running -= (samp[pos_x, pos_y] - flip_bit(samp[pos_x, pos_y])) * par[h_offset+pos_y]
        exp_running = exp(running)
        rnd = rand(Uniform(0, 1))
        if rnd < exp_running/(1+exp_running)
            samp[pos_x, pos_y] = flip_bit(samp[pos_x, pos_y])
            # and add it as a sample here after burn-in?
        end 
    end
    #samples[pos_x, :] = obs[pos_x, :] # change the row based on the sample
end 

# actually run MCMC
iter = 1000
for pos_x in [1:1:m;]
    MCMC(pos_x, iter, n, samp)
end 

# I think it works, at least it become much more extreme
mean(samp, dims=1)
mean(obs, dims=1)

# save samples 
writedlm("/home/vpoulsen/humanities-glass/analysis/data_tmp/MCMC_sim.txt", samp) # fix paths