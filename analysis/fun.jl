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