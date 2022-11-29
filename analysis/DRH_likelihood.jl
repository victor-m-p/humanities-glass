using Printf
using Statistics 
using Distributions
using DelimitedFiles

p_file = "/home/vpoulsen/humanities-glass/data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt"
mat_file = "../data/clean/matrix_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.txt"
d_main = "../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv"
sref = "../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv"
nref = "../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv"
p = np.loadtxt(p_file)

p = readdlm(p_file) 
