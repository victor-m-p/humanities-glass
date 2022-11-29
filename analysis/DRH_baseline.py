import numpy as np 
import pandas as pd 
from collections import OrderedDict
import matplotlib.pyplot as plt 

params = '../data/analysis/matrix_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.txt.mpf_params_NN1_LAMBDA0.453839'
d_main = '../data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
data = '../data/clean/matrix_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.txt'
sref = '../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
nref = '../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv'
params = np.loadtxt(params, delimiter = ',')
data = np.loadtxt(data)
d_main = pd.read_csv(d_main)
sref = pd.read_csv(sref)
nref = pd.read_csv(nref)

# our params
n, n_nan = 20, 10
h_offset = int(n*(n-1)/2) 
J = params[:h_offset] 
h = params[h_offset:]

# means, correlations 
'''
It becomes pretty complicated because we have both 
nan and weights. I want to take both into account
'''

# means 
'''
This can for sure be made smarter, 
but some of the suggestions I saw online
just ignored NAN which leads to a misleading
denominator.
'''

## split into data, weight
data_raw = np.delete(data, n, axis = 1)
data_w = data[:, n]
data_scaled = data_raw*data_w[:, None]

## col_sum (raw) and divisor (weight)
col_sum = np.sum(data_scaled, axis = 0)
divisor = np.sum(data_w)

## handling nan 
data_raw[data_raw == 0.] = np.nan
data_scaled = data_raw*data_w[:,None]
col_nan = np.count_nonzero(np.isnan(data_scaled), axis = 0)
nan_ind = np.argwhere(np.isnan(data_scaled))

### calculate nan weight per column
nan_w_dct = {} 
for nan in nan_ind: 
    row, col = nan[0], nan[1]
    w = data_w[row]
    nan_w_dct.setdefault(col,[]).append(w)
nan_w_dct.keys()
   
nan_w_dct = OrderedDict(sorted(nan_w_dct.items())) 
for i in nan_w_dct.values(): 
    print(sum(i))
nan_sum = [sum(x) for x in nan_w_dct.values()]
mean_w_nan = [x/(divisor-y) for x, y in zip(col_sum, nan_sum)]

# correlations
'''
currently I just weight it first and then I do the
correlations. I cannot find an implementation that 
satisfies both (a) weighting and (b) ignoring nan
at the same time. 
'''
d_corr = pd.DataFrame(data_scaled)
d_corr = d_corr.corr()
A_corr = d_corr.to_numpy()
m = A_corr.shape[0]
A_corr = A_corr[np.triu_indices(m, 1)]

# plot correlations 

# plot means 
d_means = pd.DataFrame({
    'raw': mean_w_nan,
    'h': h
})
d_means['index'] = d_means.index
d_means = d_means.sort_values('raw', ascending=True).reset_index(drop=True)
d_means['x'] = d_means.index
d_means['x_jit'] = [x+0.1 for x in d_means['x']]

# create figure and axis objects with subplots()
## NB: not sure about jit
fig, ax = plt.subplots(figsize = (10, 5), dpi = 300)
ax.scatter(d_means['x'],
           d_means['raw'],
           color="tab:blue")
ax.set_ylabel("Weighted Mean",
              color="tab:blue",
              fontsize=14)
ax2=ax.twinx()
ax2.scatter(d_means['x_jit'], 
            d_means['h'],
            color="tab:red",
            marker="o")
ax2.set_ylabel(r"Local fields ($h_i$)",
               color="tab:red",
               fontsize=14)
ax.tick_params(
    axis = 'x',
    which = 'both',
    bottom = False,
    top = False,
    labelbottom = False)
plt.savefig(f'../fig/h_means_nnodes_{n}_maxna_{n_nan}.pdf')

# correlations


# together
