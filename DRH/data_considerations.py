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
choices = [1, 0, -1] # how they have mostly coded it 
df['answer_types'] = np.select(conditions, choices, default="non-binary")

### this is actually really important, just keep here for now ### 
test = df[["answer_types", "answer_val", "answers", "related_q_id", "related_q", "entry_name"]]
test = test[test["answer_types"] == "non-binary"]
test.groupby('answers').size().reset_index(name="count").sort_values("count", ascending=False)


## find questions with non-binary answers and remove them
df_ = df.groupby(['related_q_id', 'answer_types']).size().reset_index(name="count")
df_ = df_[df_["answer_types"] == "non-binary"]

### notice that some of these have only a few non-binary answers ###
df_

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

