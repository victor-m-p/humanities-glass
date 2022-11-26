import pandas as pd
import numpy as np 


a, b, c, d, e = 882, 21, 20, 882, 5
df_main = pd.read_csv(f'../data/lexibank/reference/main_nrow_{a}_ncol_{b}_nuniq_{c}_suniq_{d}_maxna_{e}.csv')
A = np.loadtxt(f'../data/lexibank/clean/matrix_nrow_{a}_ncol_{b}_nuniq_{c}_suniq_{d}_maxna_{e}.txt')
df_n = pd.read_csv(f'../data/lexibank/reference/nref_nrow_{a}_ncol_{b}_nuniq_{c}_suniq_{d}_maxna_{e}.csv')
df_s = pd.read_csv(f'../data/lexibank/reference/sref_nrow_{a}_ncol_{b}_nuniq_{c}_suniq_{d}_maxna_{e}.csv')
