'''
usage: 
python DRH_curate.py -i [infile] -o [outpath] -nq [num questions] -nn [num nan]

e.g. for 20 questions, no NAN 
python DRH_curate.py -i data/DRH/raw/drh_20221019.csv -o data/DRH/clean -nq 20 -nn 0

'''

import pandas as pd 
import argparse 
from Civ import Civilizations
pd.options.mode.chained_assignment = None

def main(infile, outpath, n_questions, n_nan):
    # read data
    df = pd.read_csv(infile) # data/DRH/raw/drh_2012.json
    df = df[["entry_id", "entry_name", "related_q_id", "related_q", "related_parent_q", "answers"]] 
    # run the pipeline 
    civ = Civilizations(df)
    civ.preprocess()
    civ.set_constraints(n_questions, n_nan/n_questions, "related_q_id")
    civ.n_best()
    civ.max_constraints()
    civ.max_tolerance()
    civ.weight_format()
    civ.write_data(outpath) 
    
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--infile', required = True, type = str, help = 'path to input file (csv)')
    ap.add_argument('-o', '--outpath', required = True, type = str, help = 'path to output folder')
    ap.add_argument('-nq', '--n_questions', required = True, type = int, help = 'number of best questions')
    ap.add_argument('-nn', '--n_nan', required = True, type = int, help = 'number of nan to tolerate per civ')
    args = vars(ap.parse_args())
    main(
        infile = args['infile'],
        outpath = args['outpath'],
        n_questions = args['n_questions'],
        n_nan = args['n_nan'])