import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools

# read data
df = pd.read_csv("data/raw/df_raw.csv")
df = df[["entry_id", "entry_name", "related_q_id", "related_q", "related_parent_q", "answers"]] 

# for now outside the function
def fill_grid(df, c1, c2, fill):

    l_c1 = df[c1].drop_duplicates().to_list()
    l_c2 = df[c2].drop_duplicates().to_list()

    l_comb = list(itertools.product(l_c1, l_c2))
    d_comb = pd.DataFrame(l_comb, columns=[c1, c2])

    df = df.merge(d_comb, on = [c1, c2], how = "outer").fillna(fill)
    return df 

# sort-of works as intended now 
civ = Civilizations(df)
civ.preprocess() # can only be run once (fixed)
civ.set_constraints(20, 7/20, "related_q_id") # when we run this
civ.n_best()
civ.max_constraints()
civ.max_tolerance()
civ.create_matrix() # (180, 20)
#civ.status()
civ.save_dat("data/clean/")

pd.set_option('display.max_colwidth', None)

nref = pd.read_csv("data/clean/n_20_tol_0_nref.csv")
sref = pd.read_csv("data/clean/n_20_tol_0_sref.csv")
d = pd.read_csv("data/clean/n_20_tol_0.csv")
A = np.loadtxt("data/clean/n_20_tol_0.txt") 
d[["related_q_id"]].drop_duplicates() # 

# try with tolerance
civ = Civilizations(df)

##### CONDITION #####
## todo: only questions that are not 100% one thing
class Civilizations:
    
    def __init__(self, d, m=0, tol=0, ntot=0):
        self.d = d
        self.minv = m # value to minimize (unknown, nan) - 0 (should change...)
        self.tol = tol
        self.ntot = ntot 
        self.clean = False 
        self.sortc = False 
        self.optimc = False
        self.order_c()
    
    # create variables based on column order
    def order_c(self):
        self.sc, self.sc_, self.nc, self.nc_, self.pc, self.ac = self.d.columns
        self.check_cols()
        self.status()
         
    def check_cols(self):
        print(f"-COLUMNS-")
        print(f"samples id: {self.sc}")
        print(f"samples name: {self.sc_}")
        print(f"nodes id: {self.nc}")
        print(f"nodes name: {self.nc_}")
        print(f"parent: {self.pc}")
        print(f"answers: {self.ac}")
    
    # check status of the pipeline 
    def status(self):
        print(f'-STATUS-')
        print(f'd original (nodes, samples): ({len(self.d[self.nc].unique())}, {len(self.d[self.sc].unique())})')
        if self.clean: 
            print(f'd preprocessed (nodes, samples): ({len(self.dmain[self.nc].unique())}, {len(self.dmain[self.sc].unique())})')
        else: 
            print(f'd preprocessed: {self.clean}')
        print(f'tolerance: {self.tol}')
        print(f'total: {self.ntot}')
        print(f'sorting column: {self.sortc}')    
        print(f'optimize column: {self.optimc}')
        
    # only binary questions (used in 'preprocess' function)
    def binary(self):
        # create non-binary
        conditions = [
            (self.dmain[self.ac] == "Yes"),
            (self.dmain[self.ac] == "No"),
            (self.dmain[self.ac] == "Field doesn't know") 
            | (self.dmain[self.ac] == "I don't know") 
            | (self.dmain[self.ac] == "NaN")
        ]
        choices = [1, -1, 0]
        self.dmain[self.ac] = np.select(conditions, choices, default=100)

        # remove non-binary
        d_ = self.dmain[self.dmain[self.ac] == 100][[self.nc]].drop_duplicates()
        print(f"len non-binary: {len(d_)}")
        self.dmain = self.dmain.merge(d_, on = self.nc, how = "outer", indicator = True)
        self.dmain = self.dmain[(self.dmain._merge=="left_only")].drop("_merge", axis = 1)
        
    # preprocess function
    def preprocess(self): 
        if not self.clean:
            self.dmain = self.d[self.d[self.pc].isna()] # only top-lvl
            self.dmain = self.dmain[[self.nc, self.sc, self.ac]]
            self.dmain = self.dmain.sample(frac = 1.0, random_state=142).groupby([self.nc, self.sc]).head(1) # only one answ per q
            self.binary() # only binary answers
            self.dmain = fill_grid(self.dmain, self.sc, self.nc, 0) # fill grid with 0 
            self.dmain[self.ac] = [int(x) for x in self.dmain[self.ac]] # make better 
            self.clean = True    
        self.status()

    # right now sorts based on number of 
    def sort_vals(self): # c = specifying whether it is n (questions) or s (samples) 
        self.dsort = self.dmain.groupby([self.sortc, self.ac]).size().reset_index(name="count")
        self.dsort = self.dsort[self.dsort[self.ac] == self.minv].sort_values("count", ascending=True)  
        
    # set constraints  
    def set_constraints(self, ntot, tol, sortc):
        self.ntot = ntot 
        self.tol = tol 
        self.sortc = sortc
        if self.sortc == self.ac: 
            self.optimc = self.nc 
        elif self.sortc == self.nc: 
            self.optimc = self.sc
        else: 
            "Something went wrong here..."
        self.status() # but probably do not want the other info here...
    
    # this should probably just run sort_vals as well...
    def n_best(self): # c = self.nc
        self.sort_vals()
        self.dbest = self.dsort[[self.sortc]].head(self.ntot)
        self.dbest = self.dmain.merge(self.dbest, on = self.sortc, how = 'inner') # important
        
    # we need this to automatically be the opposite of the sorting column
    def max_constraints(self): # c = self.sc (the opposite one..)
        self.dgroup = self.dbest.groupby([self.optimc, self.ac]).size().reset_index(name = 'count')
        self.dgroup = fill_grid(self.dgroup, self.optimc, self.ac, self.minv) # ...?
        self.dgroup = self.dgroup[self.dgroup[self.ac] == self.minv].sort_values('count', ascending = True)
        self.dgroup['frac'] = self.dgroup['count']/self.ntot 
        
    def max_tolerance(self):
        #print(f"tolerance: {self.tol}")
        self.dtol = self.dgroup[self.dgroup['frac'] <= self.tol] # here?
        self.dmax = self.dbest.merge(self.dtol[[self.optimc]].drop_duplicates(), on = self.optimc, how = 'inner')
        #self.dmax.sort_values([self.sc, self.nc], ascending = [True, True], inplace = True)
        
    # create matrix 
    def create_matrix(self):
        self.dmax.sort_values([self.sc, self.nc], ascending = [True, True], inplace = True)
        d_pivot = self.dmax.pivot(
            index = self.sc, # s always index 
            columns = self.nc, # a always columns
            values = self.ac)
        self.A = np.array(d_pivot)
        print(f"A shape: {self.A.shape}")
    
    def save_dat(self, path):
        d_nobs = self.dmax[[self.nc]].drop_duplicates() # unique n
        d_sobs = self.dmax[[self.sc]].drop_duplicates() # unique s 
        d_nref = self.d[[self.nc, self.nc_]].drop_duplicates()
        d_sref = self.d[[self.sc, self.sc_]].drop_duplicates()
        self.n_ref = d_nobs.merge(d_nref, on = self.nc, how = 'inner').sort_values(self.nc, ascending = True)
        self.s_ref = d_sobs.merge(d_sref, on = self.sc, how = 'inner').sort_values(self.sc, ascending = True)
        # print information
        print(f"maximum X for {self.ntot} best X with tolerance {self.tol}")
        # save numpy
        print(f"saving array to '{path}n_{self.ntot}_tol_{self.tol}.txt'")
        np.savetxt(f"{path}n_{self.ntot}_tol_{self.tol}.txt", self.A.astype(int), fmt="%i")
        # save csv overall 
        print(f"saving csv to '{path}_n_{self.ntot}_tol_{self.tol}.csv'")
        self.dmax.to_csv(f"{path}n_{self.ntot}_tol_{self.tol}.csv", index = False)
        ## save the other ones 
        self.n_ref.to_csv(f"{path}n_{self.ntot}_tol_{self.tol}_nref.csv", index = False)
        self.s_ref.to_csv(f"{path}n_{self.ntot}_tol_{self.tol}_sref.csv", index = False)

