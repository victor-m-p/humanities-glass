using Printf
using Statistics 
using Distributions
using DelimitedFiles

# functions
function sample_params(n_nodes, scale = 1)
    n_J = Int(n_nodes*(n_nodes-1)/2)
    h_params = randn(Float64, n_nodes)
    J_params = randn(Float64, n_J)
    par = vcat(J_params, h_params) # same order as MPF_CMU
    par *= scale
    return par, n_J
end 

function get_index(n_nodes)
    ij = Array{Int64}(undef, n_nodes, n_nodes)
    cnt = 1
    for i in 1:n_nodes, j in i+1:n_nodes
        ij[i, j] = cnt
        ij[j, i] = cnt
        cnt += 1
    end 
    return ij
end

flip_bit(x) = x > 0 ? -1 : 1 

# setup
beta = 1
m = 50
n = 5

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
iter = 100
for pos_x in [1:1:m;]
    MCMC(pos_x, iter, n, samp)
end 

# I think it works, at least it become much more extreme
mean(samp, dims=1)
mean(obs, dims=1)

# save samples 
writedlm("/home/vpoulsen/humanities-glass/julia/out/samp_test.txt", samp) # fix paths