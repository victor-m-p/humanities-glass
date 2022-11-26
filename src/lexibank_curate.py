'''
usage: 
python DRH_curate.py -i [infile] -o [outpath] -nq [num questions] -nn [num nan]

e.g. for 20 questions, no NAN 
python lexibank_curate.py -i data/lexibank/raw/lexicon-values.csv -o data/lexibank/clean -nq 20 -nn 0

see lexibank_curate.sh
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
    ## id columns
    sample_id = df.groupby('Language_ID').size().reset_index(name = 'id_sample')
    sample_id['id_sample'] = sample_id.index

    node_id = df.groupby('Parameter_ID').size().reset_index(name = 'id_node')
    node_id['id_node'] = node_id.index

    df = df.merge(sample_id, on = 'Language_ID')
    df = df.merge(node_id, on = 'Parameter_ID')

    ## recode to Yes, No, Unknown
    recode_dct = {
        '?': 'Unknown',
        'True': 'Yes',
        'False': 'No'
    }
    df.replace(
        {'Value': recode_dct},
        inplace=True
    )
    df = df[['id_sample', 'Language_ID', 'id_node', 'Parameter_ID', 'Value']]

    # run the pipeline 
    civ = Civilizations(df)
    civ.preprocess() # 
    civ.set_constraints(n_questions, n_nan/n_questions, "id_node")
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
