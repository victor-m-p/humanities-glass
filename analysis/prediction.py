import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from fun import bin_states

d = pd.read_csv('../data/analysis/d_likelihood_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.csv')
sref = pd.read_csv('../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv')
nref = pd.read_csv('../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv')
d = d.merge(sref, on = 'entry_id', how = 'inner')
p = np.loadtxt('../data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt') 

# configs
n_states = 20
configs = bin_states(n_states) 

# need a way to soft match
def contains(df, str): 
    return df[df['entry_name'].str.contains(str)]

## check records
def likelihood(d, configs, cols, nref):
    d_idx, n_idx = d['p_ind'].values, len(d['p_ind'].values)
    d_conf = configs[d_idx] 
    d_mat = pd.DataFrame(d_conf, columns = cols)
    d_mat['p_ind'] = d_idx 
    d_mat = pd.melt(d_mat, id_vars = 'p_ind', value_vars = cols, var_name = 'related_q_id')
    d_mat_g = d_mat.groupby(['related_q_id', 'value']).size().reset_index(name = 'count_comb')
    d_mat_g = d_mat_g[d_mat_g['count_comb'] < n_idx]
    d_inner = d_mat.merge(d_mat_g, on = ['related_q_id', 'value'], how = 'inner')
    ref_questions = nref.merge(d_inner, on = 'related_q_id', how = 'inner')
    
    # gather 
    d_total = ref_questions.merge(d, on = 'p_ind', how = 'inner')
    d_table = pd.pivot(d_total, index = 'p_norm', columns = 'related_q', values = 'value').reset_index()
    d_table = d_table.sort_values('p_norm', ascending=False)
    return d_total, d_table 

def unknown_type(sref, nref, d_total): 
    s = d_total['entry_id'].unique()[0] # assumes only one
    row = sref[sref['s'] == s]
    l = len(row)
    row = pd.melt(row, id_vars = 's', 
                    value_vars = row.columns[1:-1],
                    var_name = 'related_q_id')
    row['related_q_id'] = row['related_q_id'].astype(int)
    
    # nan vals
    nan_vals = row[row['value'] == 0]
    nan_vals = nan_vals.rename(columns = {'value': 'type'}) 
    nan_vals = nan_vals[['related_q_id', 'type']].drop_duplicates()


    # inconsistencies
    inconst_vals = row.groupby(['related_q_id', 'value']).size().reset_index(name = 'type')
    inconst_vals = inconst_vals[inconst_vals['type'] < l]
    inconst_vals = inconst_vals[['related_q_id', 'type']].drop_duplicates()

    # merge
    all_unknown = pd.concat([nan_vals, inconst_vals])
    all_unknown = nref.merge(all_unknown, on = 'related_q_id', how = 'inner')

    return all_unknown


# other interesting records # 
# Han Confucianism
# Religion in Mesopotamia
# Irish Catholicism
# Inca
# Daoism
# Donatism (is there more than once, but treated separately)
# Northern Irish Roman Catholics
# Northern Irish Protestants
# Anglican Church
# American Evangelicalism
# Italy: Roman Catholic Christianity
# Atheism in the Soviet Union
# Religion in the Old Assyrian Period
# Religion in Judah
# Islamic modernists
# Secular Buddhists

## okay, get those with between 2 - 8 rows ## 
dn = d.groupby('entry_id').size().reset_index(name = 'count')
dn = d.merge(dn, on = 'entry_id', how = 'inner')
dsub = dn[(dn['count'] >= 2) & (dn['count'] <= 8)]
ncols = nref['related_q_id'].to_list()

# test specific ones (mostly nan)
## anglican 
anglican = contains(dsub, 'Anglican Church')
anglican_tot, anglican_q, anglican_tab = likelihood(anglican, configs, ncols, nref)
anglican_typ = unknown_type(sref, anglican_tot)

## spartan 
spartan = contains(dsub, 'Archaic Spartan Cults')
spartan_total, startan_questions, startan_table = likelihood(spartan, configs, ncols, nref)
all_unknown = unknown_type(sref, spartan_total)

## Maya
maya = contains(dsub, 'Late Classic Lowland Maya')
maya_tot, maya_q, maya_tab = likelihood(maya, configs, ncols, nref)
maya_typ = unknown_type(sref, nref, maya_tot)
maya_overview = maya_typ.merge(maya_q, on = ['related_q', 'related_q_id'], how = 'inner')

# run over all:
entry_lst = dsub['entry_id'].unique().tolist()
dct_tab = {}
dct_types = {}
for e_id in entry_lst:
    d_tmp = dsub[dsub['entry_id'] == e_id]
    d_tot, d_q, d_tab = likelihood(d_tmp, configs, ncols, nref)
    d_typ = unknown_type(sref, nref, d_tot)
    d_overview = d_typ.merge(d_q, on = ['related_q', 'related_q_id'], how = 'inner')
    dct_tab[e_id] = d_tab 
    dct_types[e_id] = d_overview

e_id = 967 
d_tmp = dsub[dsub['entry_id'] == e_id]
d_tot, d_q, d_tab = likelihood(d_tmp, configs, ncols, nref)
d_typ = unknown_type(sref, nref, d_tot)
d_overview = d_typ.merge(d_q, on = ['related_q', 'related_q_id'], how = 'inner')

d_overview
d_tab

## generally old
## spartans are fun (NA)
## 967 interesting (U Unitarians, disagreement- which makes sense)

##### other potentials ######
# Roman Divination
# Pauline Christianity
# Anglican Church
# Archaic Spartan Cults
# Gaudiya
# Late Classic Lowland Maya 
# Krishna Worship in North India - Modern Period
# !Kung
# Burmese
# Maori
# Chinese Esoteric Buddhism (Tang Tantrism)
# Hinduism in Trinidad
# Temple of the Jedi Order
# Postsocialist Mongolian Buddhism
# Orphism
# Postsocialist Mongolian Shamanism
# The Church of England
# Spartan Religion
# Unitarian Universalism
# Nestorian Christianity
# Monastic Communities of Lower Egypt: Nitria, Kellia, Scetis
# Early Christianity and Monasticism in Egypt
# Religion in Greco-Roman Egypt
# Religion in Greco-Roman Alexandria
# Atheism in the Soviet Union
# Religion in Juda
# Religion of Phonecia
# Epic of Gigamesh
# The Monastic School of Gaza
# Islamic modernists
# Book of Ezekiel
# Mesopotamian Exorcistic Texts
# Secular Buddhists
