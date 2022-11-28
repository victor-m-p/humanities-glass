'''
usage: 
python DRH_curate.py -i [infile] -o [outpath] -nq [num questions] -nn [num nan]

e.g. for 20 questions, no NAN 
python DRH_curate.py -i data/DRH/raw/drh_20221019.csv -o data/DRH/clean -nq 20 -nn 0

see DRH_curate.sh
'''

import pandas as pd 
import argparse 
from Civ import Civilizations
pd.options.mode.chained_assignment = None

def main(infile, outpath_main, outpath_ref, n_questions, n_nan):
    print(f'running N = {n_questions}, NAN tolerance = {n_nan}')
    # read data
    df = pd.read_csv(infile, low_memory=False) 
    # prep
    df = df[df['related_parent_q'].isna()] 
    df = df[["entry_id", "entry_name", "related_q_id", "related_q", "answers"]] 
    df.replace(
        {'answers': 
            {"Field doesn't know": 'Unknown', 
            "I don't know": 'Unknown'}},
        inplace = True)
    # run the pipeline 
    civ = Civilizations(df)
    civ.preprocess() # 
    civ.set_constraints(n_questions, n_nan/n_questions, "related_q_id")
    civ.n_best()
    civ.max_constraints()
    civ.max_tolerance()
    civ.weight_format()
    civ.write_data(outpath_main, outpath_ref) 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--infile', required = True, type = str, help = 'path to input file (csv)')
    ap.add_argument('-om', '--outpath_main', required = True, type = str, help = 'path to output main folder')
    ap.add_argument('-or', '--outpath_ref', required = True, type = str, help = 'path to output reference folder')
    ap.add_argument('-nq', '--n_questions', required = True, type = int, help = 'number of best questions')
    ap.add_argument('-nn', '--n_nan', required = True, type = int, help = 'number of nan to tolerate per civ')
    args = vars(ap.parse_args())
    main(
        infile = args['infile'],
        outpath_main = args['outpath_main'],
        outpath_ref = args['outpath_ref'],
        n_questions = args['n_questions'],
        n_nan = args['n_nan'])