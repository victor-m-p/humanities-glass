import pandas as pd 
import numpy as np
import itertools
from itertools import combinations, product
import os 
pd.options.mode.chained_assignment = None

# for now outside the function
def fill_grid(df, c1, c2, fill):

    l_c1 = df[c1].drop_duplicates().to_list()
    l_c2 = df[c2].drop_duplicates().to_list()

    l_comb = list(itertools.product(l_c1, l_c2))
    d_comb = pd.DataFrame(l_comb, columns=[c1, c2])

    df = df.merge(d_comb, on = [c1, c2], how = "outer").fillna(fill)
    return df 

def create_grid(df, c1, c2):

    l_c1 = df[c1].drop_duplicates().to_list()
    l_c2 = df[c2].drop_duplicates().to_list()

    l_comb = list(itertools.product(l_c1, l_c2))
    d_comb = pd.DataFrame(l_comb, columns=[c1, c2])

    return d_comb

## todo: only questions that are not 100% one thing
class Civilizations:
    
    ha = 'has_answer'
    w = 'weight'
    
    def __init__(self, d, minv=0, tol=0, ntot=0):
        self.d = d
        self.minv = minv # value to minimize (unknown, nan) - 0 (should change...)
        self.tol = tol
        self.ntot = ntot 
        self.clean = False 
        self.sortc = False 
        self.optimc = False
        self.initialize()
    
    # create variables based on column order
    def initialize(self):
        self.sc, self.sc_, self.nc, self.nc_, self.pc, self.ac = self.d.columns
        self.s_ref = self.d[[self.sc, self.sc_]].drop_duplicates()
        self.n_ref = self.d[[self.nc, self.nc_]].drop_duplicates()
    
    # check status of the pipeline 
    def status(self):
        print(f'tolerance: {self.tol}')
        print(f'number nodes: {self.ntot}')
        print(f'sorting column: {self.sortc}')    
        print(f'optimize column: {self.optimc}')
        
    # only binary questions (used in 'preprocess' function)
    def binary(self):
        # save typing 
        d = self.dmain
         
        ## first only binary answers 
        conditions = [
            (d[self.ac] == "Yes"),
            (d[self.ac] == "No"),
            (d[self.ac] == "Field doesn't know") 
            | (d[self.ac] == "I don't know") 
            | (d[self.ac] == "NaN")
        ]
        choices = [1, -1, 0]
        d[self.ac] = np.select(conditions, choices, default=100)

        # remove non-binary
        d_ = d[d[self.ac] == 100][[self.nc]].drop_duplicates() # rem., w. non-bin
        d = d.merge(d_, on = self.nc, how = "outer", indicator = True)
        d = d[(d._merge=="left_only")].drop("_merge", axis = 1)

        ### first deal with yes/no questions ###
        d_yn = d[(d[self.ac] == 1) | (d[self.ac] == -1)] 
        d_yngv = d_yn.groupby([self.nc, self.sc, self.ac]).size().reset_index(name="count")
        d_yngd = d_yn.groupby([self.nc, self.sc]).size().reset_index(name = 'count_total')
        d_yn_frac = d_yngv.merge(d_yngd, on = [self.nc, self.sc], how = 'inner')
        d_yn_frac[self.w] = d_yn_frac['count']/d_yn_frac['count_total']
        d_yn_frac = d_yn_frac[[self.nc, self.sc, self.ac, self.w]]
        
        ### then deal with nan ###
        d_yn_uniq = d_yn[[self.nc, self.sc]].drop_duplicates()
        n_s_grid = create_grid(d_yn_uniq, self.nc, self.sc)
        n_s_grid = n_s_grid.merge(d_yn_uniq, on = [self.nc, self.sc], how = 'outer', indicator = True)
        d_na_uniq = n_s_grid[(n_s_grid._merge=='left_only')].drop('_merge', axis = 1)

        ### unique cases 
        d_na_uniq[self.ha] = 0
        d_yn_uniq[self.ha] = 1
        self.duniq = pd.concat([d_na_uniq, d_yn_uniq])

        ### second case 
        d_na_frac = d_na_uniq
        d_na_frac[self.w] = 1.0
        d_na_frac = d_na_frac.rename(columns = {self.ha: self.ac})
        self.dfrac = pd.concat([d_yn_frac, d_na_frac]) 
        
    # preprocess function
    def preprocess(self): 
        if not self.clean:
            self.dmain = self.d[self.d[self.pc].isna()] # only top-lvl (still fine)
            self.binary()
            self.clean = True    

    # right now sorts based on number of 
    def sort_vals(self): # c = specifying whether it is n (questions) or s (samples) 
        self.dsort = self.duniq.groupby([self.sortc, self.ha]).size().reset_index(name="count")
        self.dsort = self.dsort[self.dsort[self.ha] == self.minv].sort_values("count", ascending=True)  
        
    # set constraints  
    def set_constraints(self, ntot, tol, sortc):
        self.ntot = ntot 
        self.tol = tol 
        self.sortc = sortc
        if self.sortc == self.sc: 
            self.optimc = self.nc 
        elif self.sortc == self.nc: 
            self.optimc = self.sc
        else: 
            "Something went wrong here..."
    
    def n_best(self): # c = self.nc
        self.sort_vals()
        self.dbest = self.dsort[[self.sortc]].head(self.ntot)
        self.dbest = self.duniq.merge(self.dbest, on = self.sortc, how = 'inner') # important
        
    # we need this to automatically be the opposite of the sorting column
    def max_constraints(self): # c = self.sc (the opposite one..)
        self.dgroup = self.dbest.groupby([self.optimc, self.ha]).size().reset_index(name = 'count')
        self.dgroup = fill_grid(self.dgroup, self.optimc, self.ha, self.minv) # this we need to check now ...
        self.dgroup = self.dgroup[self.dgroup[self.ha] == self.minv].sort_values('count', ascending = True)
        self.dgroup['frac'] = self.dgroup['count']/self.ntot 
        
    def max_tolerance(self):
        self.dtol = self.dgroup[self.dgroup['frac'] <= self.tol] # here?
        self.dmax = self.dbest.merge(self.dtol[[self.optimc]].drop_duplicates(), on = self.optimc, how = 'inner')
        self.dmax = self.dmax.merge(self.dfrac, on = [self.sc, self.nc], how = 'inner')
        self.dmax = self.dmax[[self.sc, self.nc, self.ac, self.w]].drop_duplicates()
        self.dmax.sort_values([self.sc, self.nc], ascending = [True, True], inplace = True) # maybe not needed

    def s_n_comb(self, d, N): 
        d['id'] = d.set_index([self.sc, self.nc]).index.factorize()[0]
        dct = {}
        for index, row in d.iterrows():
            id, s, n, a, w = int(row['id']), int(row[self.sc]), int(row[self.nc]), int(row[self.ac]), row['weight']
            dct.setdefault(id, []).append((s, n, a, w))
        l = list(dct.values())
        return [p for c in combinations(l, N) for p in product(*c)]

    def comb_to_df(self, comb):
        ## prepare dataframe
        vals = []
        cols = []
        for x in comb: 
            subcols = []
            subvals = []
            w_ = 1
            for y in sorted(x): 
                s, n, a, w = y 
                w_ *= w 
                # values 
                subvals.append(a)
                # columns
                subcols.append(n)
            # values
            subvals.insert(0, s)
            subvals.append(w_)
            vals.append(subvals)
            # columns 
            subcols.insert(0, 's')
            subcols.append('w')
            cols.append(subcols)
        ### make sure that format is correct
        if all((cols[i] == cols[i+1]) for i in range(len(cols)-1)):
            cols = cols[0]
        else: 
            print('inconsistent column ordering')
        self.dcsv = pd.DataFrame(vals, columns = cols)
        self.dcsv = self.dcsv.sort_values('s').reset_index(drop=True)
        self.dtxt = self.dcsv.drop(columns = 's')

    def weight_format(self): 
        
        df_lst = []
        for s in self.dmax[self.sc].unique():
            dsub = self.dmax[self.dmax[self.sc] == s]
            df_lst.append(dsub) 

        N = len(self.dmax[self.nc].unique())
        comb_lst = []
        for d in df_lst: 
            comb_lst.extend(self.s_n_comb(d, N))
        self.temp = comb_lst

        self.comb_to_df(comb_lst)
    
    def write_data(self, mainpath, refpath):
        # extract unique s and n 
        s_uniq = self.dmax[[self.sc]].drop_duplicates()
        n_uniq = self.dmax[[self.nc]].drop_duplicates()
        s_out = self.s_ref.merge(s_uniq, on = self.sc, how = 'inner').sort_values(self.sc)
        n_out = self.n_ref.merge(n_uniq, on = self.nc, how = 'inner').sort_values(self.nc)
        # get information for writing 
        nrow, ncol = self.dtxt.shape
        stot = len(s_uniq)
        tol = int(self.tol * self.ntot)
        # write files 
        identifier = f"nrow_{nrow}_ncol_{ncol}_nuniq_{self.ntot}_suniq_{stot}_maxna_{tol}"
        csv_outname = os.path.join(refpath, f'main_{identifier}.csv')
        txt_outname = os.path.join(mainpath, f'matrix_{identifier}.txt')
        s_outname = os.path.join(refpath, f'sref_{identifier}.csv')
        n_outname = os.path.join(refpath, f'nref_{identifier}.csv')
        #print(f'writing main to folder {mainpath} with name {identifier}')
        #print(f'writing references to folder {refpath} with name {identifier}')
        self.dcsv.to_csv(csv_outname, index = False)
        self.dtxt.to_csv(txt_outname, sep = ' ', header = False, index = False)
        s_out.to_csv(s_outname, index = False) 
        n_out.to_csv(n_outname, index = False)