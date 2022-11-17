import itertools
import pandas as pd 

def fill_grid(df, c1, c2, fill): 
    """All combinations of c1-c2. Fill missing combinations of c1-c2 in df with NaN (string)

    Args:
        df (pd.DataFrame): dataframe with columns c1, c2
        c1 (string): column in df 
        c2 (string): column in df 

    Returns:
        pd.Dataframe: dataframe with non-present combinations of c1-c2 filled with NaN
    """
    l_c1 = df[c1].drop_duplicates().to_list()
    l_c2 = df[c2].drop_duplicates().to_list()

    l_comb = list(itertools.product(l_c1, l_c2))
    d_comb = pd.DataFrame(l_comb, columns=[c1, c2])

    df = df.merge(d_comb, on = [c1, c2], how = "outer").fillna(fill)

    return df

def assign_id(df, c_group, c_new):
    """create new id column in df

    Args:
        df (pd.DataFrame): Dataframe with column c_group
        c_group (string): column in df
        c_new (string): new id column in df

    Returns:
        pd.DataFrame: Dataframe with new id column c_new
    """
    d_id = df.groupby(c_group).size().reset_index(name="count").sort_values("count", ascending = False)
    d_id[c_new] = d_id.index
    d_id = d_id[[c_group, c_new]]
    df = df.merge(d_id, on = c_group, how = "inner") 
    return df

def calc_nodes_samples(d, c_q_id, c_ent_id, c_ans, s_minval, node_lst, sample_lst):
    """returns one row for each (n_node, n_sample) combination with fraction NA.

    Args:
        d (pd.DataFrame): DataFrame with columns c_q_id, c_ent_id, c_ans
        c_q_id (string): c_q_id (column in d with questions/nodes)
        c_ent_id (string): column in d with entries/samples
        c_ans (string): column in d with answer type (e.g. NA, Yes, No)
        s_minval (string): the answer type to minimize
        node_lst (list): list of integers (number of nodes)
        sample_lst (list): list of integers (number of samples)

    Returns:
        pd.DataFrame: DataFrame with all combinations of elem. from node_lst, sample_lst and fraction of NA.
    """
    l = []
    
    ## new try: first sort questions by number of NaN
    df_at = d.groupby([c_q_id, c_ans]).size().reset_index(name = "count")
    df_at = df_at[df_at[c_ans] == s_minval].sort_values("count", ascending=True)

    ## then select N questions (nodes) with most answers
    for n_nodes in node_lst: # number of nodes 
        top_q = df_at[[c_q_id]].head(n_nodes) 
        tst = d.merge(top_q, on = c_q_id, how = "inner")

        ## then sort by entries 
        tst2 = tst.groupby([c_ent_id, c_ans]).size().reset_index(name = "count")
        tst3 = fill_grid(tst2, c_ent_id, c_ans, 0)
        tst3 = tst3[tst3[c_ans] == s_minval].sort_values("count", ascending=True)

        ## then select N civilizations (samples) with most answers
        for n_samples in sample_lst: # number of samples
            tst4 = tst3.head(n_samples)

            ## then calculate fraction of NA
            frac_na = tst4["count"].sum() / (n_samples*n_nodes) 
            l.append((n_nodes, n_samples, frac_na))
    
    # gather in dataframe
    d = pd.DataFrame(l, columns = ["n_nodes", "n_samples", "frac_na"])
    return d 


def single_size(df, c1, ascending = False): 
    return df.groupby(c1).size().reset_index(name = "count").sort_values("count", ascending = ascending)

def groupby_size(df, c1, c2, ascending = False): 
    df_grouped = df.groupby([c1, c2]).size().reset_index(name = 'count').sort_values('count', ascending = ascending)
    return df_grouped 

def distinct_size(df, c1, c2, ascending = False): 
    df = df[[c1, c2]].drop_duplicates()
    d_c1 = df.groupby(c1).size().reset_index(name = 'count').sort_values("count", ascending = ascending)
    d_c2 = df.groupby(c2).size().reset_index(name = 'count').sort_values("count", ascending = ascending)
    return d_c1, d_c2 