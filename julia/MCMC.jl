using Printf
using Statistics 


beta = 1
m = 50
n = 5

function sample_params(n_nodes, scale = 1)
    n_J = Int(n_nodes*(n_nodes-1)/2)
    h_params = randn(Float64, n_nodes)
    J_params = randn(Float64, n_J)
    par = vcat(J_params, h_params) # same order as MPF_CMU
    par *= scale
    return par 
end 

randn()

par = sample_params(n) # big list 
obs = rand([-1, 1], m, n) # obs 

# MCMC 
## loop over each row in observations 
## pick random point (for I in iter) -- so in (1, n_nodes?)
flip_bit(x) = x > 0 ? -1 : 1 
function flip_index(x, i) # would be really nice to do this in one line..
    y = deepcopy(x)
    val = flip_bit(y[i])
    y[i] = val
    return y 
end  

par

flip_bit(obs[pos_x, pos_y])
obs[pos_x, pos_y]

pos_y = rand(pos_n)
pos_x = 1 # config 
iter = 10
for i in [1:1:iter;]
    pos_y = rand([1:1:n;])
    running = 0
    for j in [1:1:n;]
        if j != pos_y 
            println(j)
            running += (obs[pos_x, pos_y] - flip_bit(obs[pos_x, pos_y])) * obs[pos_x, j] * par[***]
        end 
    end 
    running += (obs[pos_x, pos_y] - flip_bit(obs[pos_x, pos_y])) * par[h_offset+pos_y] 
end 