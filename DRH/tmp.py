import pandas as pd 

# setup
n_rows, n_nan, n_nodes, n_top_states = 455, 5, 20, 150

d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')


len(d_likelihood)
d_likelihood.head(5)