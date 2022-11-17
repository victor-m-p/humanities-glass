import pandas as pd 
from tqdm import tqdm
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import permutations

df = pd.read_csv("../data/df_raw.csv")

def groupby_size(df, c1, c2, ascending = False): 
    df_grouped = df.groupby([c1, c2]).size().reset_index(name = 'count').sort_values('count', ascending = ascending)
    return df_grouped 

def distinct_size(df, c1, c2, ascending = False): 
    df = df[[c1, c2]].drop_duplicates()
    d_c1 = df.groupby(c1).size().reset_index(name = 'count').sort_values("count", ascending = ascending)
    d_c2 = df.groupby(c2).size().reset_index(name = 'count').sort_values("count", ascending = ascending)
    return d_c1, d_c2 

# one poll per entry id 
entid_by_poll, _ = distinct_size(df, "entry_id", "poll")
entid_by_poll # each entry id occurs with just one poll
# several polls per entry name (makes sense)
entname_by_poll, _ = distinct_size(df, "entry_name", "poll")
entname_by_poll
yahgan = df[df["entry_name"] == "Yahgan"]
# both yahgan are religious GROUP (v5, v6)
yahganv5 = yahgan[yahgan["poll"] == "Religious Group (v5)"]
yahganv6 = yahgan[yahgan["poll"] == "Religious Group (v6)"]
# are these the same? 


# e.g. https://religiondatabase.org/browse/search?q=Yahgan
# does not have same dates (so we would fail to match)
# 


# mapping 
## region_tags
regtags_per_entry, _ = distinct_size(df, "entry_id", "region_tags")
regtags_per_entry # not unique, i.e. one rel can have conflicting tags 
## test a couple 
dfregt = df[df["entry_id"] == 1324]
dfregt["entry_name"]
dfregt["region_tags"].unique() # but says "People's Republic of China"
### Q: more specifically, which one do we match on (or all?)
#### e.g.: Asia[1] (too high level?)
#### e.g.: China[504] (maybe good?)
#### e.g.: Gansu[757] (too specific?)

## entry_tags 
enttags_per_entry, _ = distinct_size(df, "entry_id", "entry_tags")
enttags_per_entry # this is unique 
df["region_tags"] # region tags, way to combine
df[["region_desc", "region_tags"]].head(5)

## time (year)--??
## those with more than one end_year
year_per_entry, _ = distinct_size(df, "entry_id", "end_year")
## test a couple 
year_per_entry
dfend = df[df["entry_id"] == 1324]
dfend["entry_name"]
dfend["end_year"].max()
dfend["start_year"].min()
dfend["start_year"].unique()
# matches daterange for entry_id = 197 (i.e. date range)
# matches daterenge for entry_id = 1180
# does not match daterange for entry id = 1231: ...?
# matches end year for entry_id = 1324 (but we get -400 for start_date, but says 150CE).
### Q: to match, are we doing date range (start-end)?
### CE = AD?
