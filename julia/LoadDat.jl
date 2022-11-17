using DelimitedFiles
using Statistics

# http://julianlsolvers.github.io/Optim.jl/v0.9.3/user/minimization/
# https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/

# assume that we know n, m 
n = 3
m = 100

# crazy that this just works params
p_true = readdlm("religion-glass/julia/sim_data/hJ_nodes_3_samples_100_scale_0.1.txt", Float64)
samp = readdlm("religion-glass/julia/sim_data/samples_nodes_3_samples_100_scale_0.1.txt", Int64)

# unique states 
uniq = unique(samp, dims = 1)
uniq = [uniq[i,:] for i in 1:size(uniq,1)] # clean this 

# 
n_param = Int16(n*(n-1)/2+n)
h_off = Int(n*(n-1)/2)

# neighbors for each
## there must be a better way 
## also, how do we extend this to number of neighbors > 1 
flip_bit(x) = x > 0 ? -1 : 1 
function flip_index(x, i) # would be really nice to do this in one line..
    y = deepcopy(x)
    val = flip_bit(y[i])
    y[i] = val
    return y 
end  

# get unique 1_flip neighbors (not sure how to generalize yet)
nuniq = [[flip_index(row, i) for i in eachindex(row)] for row in uniq]

# now we basically do the math...?
nuniq
size(nuniq)
size(uniq)
nuniq[1][1] # I can index in, but probably cannot take len 

# compute k 
dk = [0.0 for i in 1:n_param]
ei = [0.0 for i in 1:length(uniq)]
nei = [[0.0 for i in 1:n] for j in 1:length(nuniq)]

uniq[1]
for (index, elem) in enumerate(uniq)
    println("index $index\n")
    println("elem $elem\n")
end 

# loop over stuff 
# we can probably compress this 
for (d, row) in enumerate(uniq)
    p_n = 1
    for i in 1:n 
        for j in i+1:n 
            ei[d] += row[i]*row[j]*p_true[p_n]  
            p_n += 1
        end 
        ei[d] += row[i]*p_true[p_n]
    end 
    ei[d] *= -1 # negative 1 
end 

# The function for nearest neighbors from Simon is 
# outdated. Have to look at the new version 


# can we make the smart loop?
println("start")
for i in 1:n, j in i+1:n
    println("i $i")
    println("j $j")
end 
