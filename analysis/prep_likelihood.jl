# todo: add entry_name to the dataframe 
# would save a lot of wrangling. 
using Printf, Statistics, Distributions, DelimitedFiles, CSV, DataFrames, IterTools

# check up on how to better manage paths in julia
p_file = "/home/vpoulsen/humanities-glass/data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt"
all_file = "/home/vpoulsen/humanities-glass/data/analysis/allstates_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt"
unweighted_file = "/home/vpoulsen/humanities-glass/data/analysis/d_collapsed_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.csv"

# setup
n_nodes, maxna = 20, 10

# read stuff
p = readdlm(p_file) 
allstates = readdlm(all_file)
d_unweighted = DataFrame(CSV.File(unweighted_file))
mat_unweighted = Matrix(d_unweighted)

# quick functions
f(iterators...) = vec([collect(x) for x in product(iterators...)])
matchrow(a,B) = findfirst(i->all(j->a[j] == B[i,j],1:size(B,2)),1:size(B,1))

## expand nan
function expand_nan(l)
    lc = [x == 0 ? [1, -1] : [x] for x in l]
    return return f(lc...)
end 

## main function
function get_ind(data_state, allstates, n_nodes)
    v_ind = Vector{Int64}(undef, 0)
    m_obs = Matrix{Int64}(undef, 0, n_nodes)
    for i in data_state
        m = reshape(i, 1, length(i))
        hit = matchrow(m, allstates)
        v_ind = [v_ind;hit]
        m_obs = [m_obs;m]
    end 
    return v_ind, m_obs
end 

## the major loop
rows, cols = size(mat_unweighted)
total_states = Matrix{Int64}(undef, 0, n_nodes)
total_praw = Vector{Float64}(undef, 0)
total_pnorm = Vector{Float64}(undef, 0)
total_entry = Vector{Int64}(undef, 0)
total_pind = Vector{Int64}(undef, 0)
for i in [1:1:rows;] 
    println(i)
    entry_id = mat_unweighted[i,1]
    vals = mat_unweighted[i:i, 2:cols]
    data_state = expand_nan(vals)
    p_index, obs_states = get_ind(data_state, allstates, n_nodes)
    p_raw = p[p_index]
    p_norm = p_raw./sum(p_raw, dims = 1)

    # tracking id 
    l = size(p_norm)[1]
    entry_vec = fill(entry_id, l)

    # append stuff
    global total_states = [total_states;obs_states]
    global total_praw = [total_praw;p_raw]
    global total_pnorm = [total_pnorm;p_norm] 
    global total_entry = [total_entry;entry_vec]
    global total_pind = [total_pind;p_index]
end

# final data  
mat = hcat(total_entry, total_states)
total_pind = total_pind .- 1 # for 0-indexing in python
d = DataFrame(
    entry_id = total_entry,
    p_ind = total_pind, 
    p_raw = total_praw, 
    p_norm = total_pnorm)

# save stuff 
CSV.write("/home/vpoulsen/humanities-glass/data/analysis/d_likelihood_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.csv", d)
writedlm("/home/vpoulsen/humanities-glass/data/analysis/mat_likelihood_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt", mat)