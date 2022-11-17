import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import *

# read data
df = pd.read_csv("data/raw/df_raw.csv")

##### BASIC PREPROCESSING #####
## subset relevant columns
df = df[["related_q", "related_q_id", "answers", "answer_val", "related_parent_q", "entry_name", "entry_id"]]
## only overall questions (without parent)
df = df[df["related_parent_q"].isna()]
## if more than one answer to a question sample only 1 answer. 
## maybe this should be later actually 
df = df.sample(frac = 1.0).groupby(['related_q_id', 'entry_id']).head(1)
## check for yes/no questions 
conditions = [
    (df['answers'] == "Yes"),
    (df["answers"] == "No"),
    (df['answers'] == "Field doesn't know") | (df["answers"] == "I don't know") | (df['answers'] == "NaN")
]
choices = ["Yes", "No", "DK-NAN"]
df['answer_types'] = np.select(conditions, choices, default="non-binary")

df.groupby('answer_types').size()

## find questions with non-binary answers and remove them
df_ = df.groupby(['related_q_id', 'answer_types']).size().reset_index(name="count")
df_ = df_[df_["answer_types"] == "non-binary"]
df_ = df.merge(df_, how = "outer", indicator = True)
df_b = df_[(df_._merge=="left_only")].drop("_merge", axis = 1)

## fill with nan 
df_nan = fill_grid(df_b, "related_q_id", "entry_id", "DK-NAN")

## recode values
answer_map = {
    "DK-NAN": 0,
    "Yes": 1,
    "No": -1
}
df_nan = df_nan.assign(answers = df_nan["answer_types"].map(answer_map))
df_nan = df_nan[["entry_id", "related_q_id", "answers"]]

##### END BASIC PREPROCESSING ######

##### CONDITION #####
## only questions that are not 100% one thing
class civs:
    def __init__(self, d, n, s, a, m):
        self.d = d # basic data
        self.n = n # questions column (nodes) - related_q_id
        self.s = s # samples column (civilization) - entry_id
        self.a = a # answer column (e.g. Yes = 1, No = -1, DK/NAN = 0) - answers
        self.m = m # value to minimize (unknown, nan) - 0

    def sort_n(self):
        d_n = self.d.groupby([self.n, self.a]).size().reset_index(name="count")
        d_n = d_n[d_n[self.a] == self.m].sort_values("count", ascending=True)
        self.d_n = d_n 

    def sort_s(self): 
        d_s = self.d.groupby([self.s, self.a]).size().reset_index(name="count")
        d_s = d_s[d_s[self.a] == self.m].sort_values("count", ascending=True)
        self.d_s = d_s
    
    def n_best(self, tol, nq):
        # get the best N nodes (questions)
        d_ = self.d_n[[self.n]].head(nq) # len n_q
        # inner join with data
        d_top = self.d.merge(d_, on = self.n, how = 'inner')
        # samples answer distribution for best N nodes 
        d_ = d_top.groupby([self.s, self.a]).size().reset_index(name = 'count')
        # fill grid if incomplete 
        d_ = fill_grid(d_, self.s, self.a, self.m) # just fill zeros
        # sort values by answer to minimize (DK/NAN) 
        d_ = d_[d_[self.a] == self.m].sort_values('count', ascending = True)
        # assign fraction for when we have tolerance > 0
        d_['frac'] = d_['count']/nq
        # take all samples with lower DK/NAN than tolerance
        d_ = d_[d_['frac'] <= tol]
        d_ = d_top.merge(d_[[self.s]].drop_duplicates(), on = self.s, how = "inner")
        d_.sort_values([self.s, self.n], ascending = [True, True], inplace = True)
        self.d_n_t = d_
    
    def s_best(self, tol, nc):
        # get best N samples (civs)
        dx = self.d_s[[self.s]].head(nc) # 200
        # inner join with data
        dx_top = self.d.merge(dx, on = self.s, how = 'inner')
        # 
        dx = dx_top.groupby([self.n, self.a]).size().reset_index(name = 'count')
        # fill grid (just zeros filled, so not complete)
        dx = fill_grid(dx, self.n, self.a, self.m)
        #
        dx = dx[dx[self.a] == self.m].sort_values('count', ascending = True)
        # assign frac 
        dx['frac'] = dx['count']/nc
        # take values out 
        dx = dx[dx['frac'] <= tol] # n = 46
        dx = dx_top.merge(dx[[self.n]].drop_duplicates(), on = self.n, how = "inner")
        dx.sort_values([self.s, self.n], ascending = [True, True], inplace = True)
        self.d_s_t = dx
    
    def create_mat(self, tol, nq):
        self.sort_n()
        self.n_best(tol, nq)
        d_pivot = self.d_n_t.pivot(
            index = self.s,
            columns = self.n,
            values = self.a)
        self.A = np.array(d_pivot)
    
    def save_dat(self, path, tol, nq):
        # need somehow to test whether this already happened
        # i.e. can we check whether it exists?
        self.create_mat(tol, nq)
        
        #x = self.d.merge(self.d_n_t, on = "related_q_id", how = "inner")
        #x = x[["related_q_id", "related_q"]].drop_duplicates()
        # save stuff
        x.to_csv(f"{path}n_20_tol_0_q.csv", index = False)
        np.savetxt(f"{path}n_20_tol_0.txt", self.A.astype(int), fmt="%i")
        self.d_n_t.to_csv(f"{path}n_20_tol_0.csv", index = False)

civdat = civs(df_nan, "related_q_id", "entry_id", "answers", 0)
civdat.save_dat("data/clean/", 0, 20)

# load it (looks good)
d = pd.read_csv("data/clean/n_20_tol_0.csv")
A = np.loadtxt("data/clean/n_20_tol_0.txt", dtype = "int")
x = d.merge(df, on = "related_q_id", how = "inner")[["related_q_id", "related_q"]].drop_duplicates()
x.to_csv("data/clean/n_20_tol_0_q.csv", index = False)

# clean this more and make the class better
# test everything a lot ...
# make sure that Simon has the right stuff