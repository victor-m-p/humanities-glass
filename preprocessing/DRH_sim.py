'''
usage: 
python DRH_sim.py -np 5 -ns 10000 -sh 1 -sj 1 -o data/DRH/sim
'''

import numpy as np 
from sim_fun import p_dist, bin_states
import argparse 
import os 

def hJ(n_nodes, scale_h, scale_J): 
    h = np.random.normal(scale=scale_h, size=n_nodes)
    J = np.random.normal(scale=scale_J, size=n_nodes*(n_nodes-1)//2)
    return h, J

def sample_data(n_nodes, h, J, n_samples): 
    # sample based on J, h
    p = p_dist(h, J) # probability of all states
    allstates = bin_states(n_nodes)
    sample = allstates[np.random.choice(range(2**n_nodes), # doesn't have to be a range
                                        size=n_samples, # how many samples
                                        replace=True, # a value can be selected multiple times
                                        p=p)] 
    return sample 

def corr_mean(sample): 
    # get raw correlations and raw mean 
    ## does not make a difference for corr whether
    ## it is coded as -1 or 0 I think. 
    ## can we get correlations directly from params?
    corr = np.corrcoef(sample.T)
    m = corr.shape[0]
    corr = corr[np.triu_indices(m, 1)]
    means = np.mean(sample, axis = 0)
    return corr, means 

def main(n_nodes, n_samples, scale_h, scale_J, outpath): 
    seed = 124
    np.random.seed(seed)
    h, J = hJ(n_nodes, scale_h, scale_J)
    sample = sample_data(n_nodes, h, J, n_samples)
    corr, means = corr_mean(sample)
    outstring = f'nnodes_{n_nodes}_hscale_{scale_h}_Jscale_{scale_J}_nsamp_{n_samples}_seed_{seed}.txt'
    outnames = ['h', 'J', 'corr', 'means', 'samp']
    outvars = [h, J, corr, means, sample]
    for outname, outvar in zip(outnames, outvars): 
        tmp_path = os.path.join(outpath, f"{outname}_{outstring}")
        np.savetxt(tmp_path, outvar)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-np', '--n_nodes', required = True, type = int, help = 'number nodes')
    ap.add_argument('-ns', '--n_samples', required = True, type = int, help = 'number samples')
    ap.add_argument('-sh', '--scale_h', required = True, type = float, help = 'scale for local fields')
    ap.add_argument('-sj', '--scale_J', required = True, type = float, help = 'scale for pairwise couplings')
    ap.add_argument('-o', '--outpath', required = True, type = str, help = 'outpath')
    args = vars(ap.parse_args())
    main(
        n_nodes = args['n_nodes'],
        n_samples = args['n_samples'],
        scale_h = args['scale_h'],
        scale_J = args['scale_J'],
        outpath = args['outpath'])