


######## old stuf #######

# clean this more and make the class better
# test everything a lot ...
# make sure that Simon has the right stuff
# make sure that all questions different kinds of answers
class civs:

    def __init__(self, d, n, s, a, m):
        self.d = d # basic data
        self.n = n # questions column (nodes) - related_q_id
        self.s = s # samples column (civilization) - entry_id
        self.a = a # answer column (e.g. Yes = 1, No = -1, DK/NAN = 0) - answers
        self.m = m # value to minimize (unknown, nan) - 0

    # some duplicate nature here (sort_n, sort_s)
    def sort_n(self):
        print(f"-1. len d: {len(self.d)}")
        d_n = self.d.groupby([self.n, self.a]).size().reset_index(name="count")
        print(f"0. len d_s: {len(d_n)}")
        d_n = d_n[d_n[self.a] == self.m].sort_values("count", ascending=True)
        self.d_n = d_n 
        print(f"1. len d_s: {len(d_n)}")

    def sort_s(self): 
        d_s = self.d.groupby([self.s, self.a]).size().reset_index(name="count")
        d_s = d_s[d_s[self.a] == self.m].sort_values("count", ascending=True)
        self.d_s = d_s
    
    # ... should only be one of the ... # 
    
    # some duplicate nature here (n_best, s_best)
    def n_best(self, tol, nq):
        d_ = self.d_n[[self.n]].head(nq)
        print(f"2. len d_: {len(d_)}")
        d_top = self.d.merge(d_, on = self.n, how = 'inner')
        print(f"3. len d_top: {len(d_top)}")
        d_ = d_top.groupby([self.s, self.a]).size().reset_index(name = 'count')
        print(f"4. len d_: {len(d_)}")
        d_ = fill_grid(d_, self.s, self.a, self.m)
        print(f"5. len d_: {len(d_)}")
        d_ = d_[d_[self.a] == self.m].sort_values('count', ascending = True)
        print(f"6. len d_: {len(d_)}")
        d_['frac'] = d_['count']/nq
        print(f"7. len d_: {len(d_)}")
        d_ = d_[d_['frac'] <= tol]
        print(f"8. len d_: {len(d_)}") 
        d_ = d_top.merge(d_[[self.s]].drop_duplicates(), on = self.s, how = "inner")
        d_.sort_values([self.s, self.n], ascending = [True, True], inplace = True)
        self.d_n_t = d_

    def s_best(self, tol, nc):
        dx = self.d_s[[self.s]].head(nc)
        dx_top = self.d.merge(dx, on = self.s, how = 'inner')
        dx = dx_top.groupby([self.n, self.a]).size().reset_index(name = 'count')
        dx = fill_grid(dx, self.n, self.a, self.m)
        dx = dx[dx[self.a] == self.m].sort_values('count', ascending = True)
        dx['frac'] = dx['count']/nc
        dx = dx[dx['frac'] <= tol]
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
        #x.to_csv(f"{path}n_20_tol_0_q.csv", index = False)
        np.savetxt(f"{path}n_20_tol_0.txt", self.A.astype(int), fmt="%i")
        self.d_n_t.to_csv(f"{path}n_20_tol_0.csv", index = False)

# read data
df = pd.read_csv("data/raw/df_raw.csv")
##### BASIC PREPROCESSING #####
## subset relevant columns
df = df[["related_q", "related_q_id", "answers", "answer_val", "related_parent_q", "entry_name", "entry_id"]]
## only overall questions (without parent)
df = df[df["related_parent_q"].isna()]
## if more than one answer to a question sample only 1 answer. 
## maybe this should be later actually 
df = df.sample(frac = 1.0, random_state=142).groupby(['related_q_id', 'entry_id']).head(1)
## check for yes/no questions 
conditions = [
    (df['answers'] == "Yes"),
    (df["answers"] == "No"),
    (df['answers'] == "Field doesn't know") | (df["answers"] == "I don't know") | (df['answers'] == "NaN")
]
choices = ["Yes", "No", "DK-NAN"] # how they have mostly coded it 
df['answer_types'] = np.select(conditions, choices, default="non-binary")

## find questions with non-binary answers and remove them
df_ = df.groupby(['related_q_id', 'answer_types']).size().reset_index(name="count")
df_ = df_[df_["answer_types"] == "non-binary"]
#df_cp = df_
#df_cp
#len(df_) # 54704 -- but this goes wrong somehow...?
df_x = df.merge(df_, how = "outer", indicator = True)
#len(df_x[df_x['_merge'] == 'left_only']) #50705
#df_left = df_x[df_x['_merge'] == 'left_only']
#df_left.groupby('related_q_id').size() # 156
#len(df_) # 54704
df_b = df_x[(df_x._merge=="left_only")].drop("_merge", axis = 1)
#x = df[df["related_q_id"] == 4698]
#x.groupby('answer_types').size() # it is wrong

#len(df_b) # 50704
## fill with nan 
df_nan = fill_grid(df_b, "related_q_id", "entry_id", "DK-NAN")
#df_b.groupby('answer_types').size()
## recode values
answer_map = {
    "DK-NAN": 0,
    "Yes": 1,
    "No": -1
}
df_nan = df_nan.assign(answers = df_nan["answer_types"].map(answer_map))
#df_nan.groupby('answer_types').size() # ...
df_nan = df_nan[["entry_id", "related_q_id", "answers"]]
#len(df_nan) # 130416 
#df_nan[df_nan["related_q_id"] == 4698] # just all 0, so why does it make a difference?
#df_nan.head(5)
#df_nan.groupby('answers').size() 
# -1: 22664 -- a few more  
#  0: 85735 -- many more here
#  1: 22017 --- a few more
len(df_nan)
x = civs(df_nan, n = 'related_q_id', s = 'entry_id', a = 'answers', m = 0)
x.create_mat(tol=0, nq=20)
x.A.shape # (180, 20)--now also (178, 20)??
ydnt = x.d_n_t
ydnt
## why is it not the same as this????


### the alternative pipeline ###
df = pd.read_csv("data/raw/df_raw.csv")
##### BASIC PREPROCESSING #####
## subset relevant columns
df = df[["related_q", "related_q_id", "answers", "answer_val", "related_parent_q", "entry_name", "entry_id"]]
## only overall questions (without parent)
df = df[df["related_parent_q"].isna()]
## if more than one answer to a question sample only 1 answer. 
## maybe this should be later actually 
df = df.sample(frac = 1.0, random_state = 142).groupby(['related_q_id', 'entry_id']).head(1)

## check for yes/no questions 
conditions = [
    (df['answers'] == "Yes"),
    (df["answers"] == "No"),
    (df['answers'] == "Field doesn't know") | (df["answers"] == "I don't know") | (df['answers'] == "NaN")
]
choices = [1, -1, 0] # how they have mostly coded it 
df['answers'] = np.select(conditions, choices, default=100)

## find questions with non-binary answers and remove them
d_ = df[df["answers"] == 100][["related_q_id"]].drop_duplicates()

#len(d_) # 22 
#d_.sort_values('related_q_id')
#df_cp.sort_values('related_q_id')
#d_
dfx = df.merge(d_, on = "related_q_id", how = "outer", indicator = True)
#len(dfx) # 54704
dfy = dfx[(dfx._merge=="left_only")].drop("_merge", axis = 1) # here we differ
dfy
#len(dfy) # 49951 - and there it is
#len(dfy[dfy['_merge'] == 'left_only']) # 49951
#len(df_x[df_x['_merge'] == 'left_only']) # 50705
#df_x_left = df_x[df_x['_merge'] == 'left_only']
#dfy_left = dfy[dfy['_merge'] == 'left_only']
#df_x_ls = df_x_left.groupby('related_q_id').size().reset_index(name = 'count')
#dfy_lls = dfy_left.groupby('related_q_id').size().reset_index(name = 'count')
#df_x_ls # more 
#dfy_lls # fewer

#test = pd.concat([dfy_lls, df_x_ls]).drop_duplicates(keep=False)
#test
#d_.sort_values(by = 'related_q_id')
#dfy_lls[dfy_lls["related_q_id"] == 4698] # new one does not
#df_x_ls[df_x_ls["related_q_id"] == 4698] # this one has it
#df_x_left[df_x_left["related_q_id"] == 4698] # yep, is there...
#dfy_left[dfy_left["related_q_id"] == 4698] # not there

## what are these questions? ##
#df[df["related_q_id"] == 4699]

## fill with nan 
dfy.dtypes # answers is int64
df_nanx = fill_grid(dfy, "related_q_id", "entry_id", 0)

df_nanx.groupby('answers').size()
df_nanx["answers"] = [int(x) for x in df_nanx["answers"]] 
# zero becomes different things, omg. 
# just run the whole new pipeline without class
len(df_nanx)
df_nanx = df_nanx[["entry_id", "related_q_id", "answers"]]
x = civs(df_nanx, n = 'related_q_id', s = 'entry_id', a = 'answers', m = 0)
x.create_mat(tol=0, nq=20)
x.A.shape # (180, 20) ---slightly smaller..., how?
dntx = x.d_n_t

# why are they not the same?
# now they ARE the same 
pd.concat([ydnt, dntx]).drop_duplicates(keep=False)




###### main problem ########
## what we have ##
d1 = pd.DataFrame({
    'entry_id': [174, 174, 174, 174, 174, 174, 175, 175, 175, 175],
    'related_q_id': [400, 500, 600, 600, 500, 700, 400, 500, 600, 700],
    'answers': [1, 1, 1, -1, -1, 1, 1, -1, 1, 1],
    'weight': [1, 0.7, 0.5, 0.5, 0.3, 1, 1, 1, 1, 1]
})

def s_n_comb(d, N): 
    d['id'] = d.set_index(['entry_id','related_q_id']).index.factorize()[0]
    dct = {}
    for index, row in d.iterrows():
        id, s, n, a, w = int(row['id']), int(row['entry_id']), int(row['related_q_id']), int(row['answers']), row['weight']
        dct.setdefault(id, []).append((s, n, a, w))
    l = list(dct.values())
    return [p for c in combinations(l, N) for p in product(*c)]

def comb_to_df(comb):
    ## prepare dataframe
    vals = []
    cols = []
    for x in comb: 
        subcols = []
        subvals = []
        w_ = 1
        for y in x: 
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
    dx = pd.DataFrame(vals, columns = cols)
    dxx = dx.drop(columns = 's')
    return dx, dxx

def weight_format(d1): 
    
    df_lst = []
    for s in d1['entry_id'].unique():
        dsub = d1[d1['entry_id'] == s]
        df_lst.append(dsub) 

    N = len(d1['related_q_id'].unique())
    comb_lst = []
    for d in df_lst: 
        comb_lst.extend(s_n_comb(d, N))
    comb_lst

    dx, dxx = comb_to_df(comb_lst)
    return dx, dxx
