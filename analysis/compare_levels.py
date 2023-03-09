'''
Compare constraints at the level of parameters and configurations.
Relies on calculate_pairs.py and calculate_params.py
'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

def compare_hi(d, x, xlab, y1, y1lab, y2, y2lab, idx_list=None, outname=None): 
    # calculate axis limits
    xmin_y1 = d[y1].min()
    xmax_y1 = d[y1].max()
    xmin_y2 = d[y2].min()
    xmax_y2 = d[y2].max()
    y1_range = np.sum(np.abs([xmin_y1, xmax_y1]))/8
    y2_range = np.sum(np.abs([xmin_y2, xmax_y2]))/8
    
    # plot 
    fig, ax1 = plt.subplots()
    ax1_ylabel = ax1.set_ylabel(y1lab)
    ax1_ylabel.set_color('tab:blue')
    ax1_ylabel.set_size(14)
    ax1.set_ylim(xmin_y1-y1_range, xmax_y1+y1_range)
    ax1.scatter(d[x], 
                d[y1], 
                color="tab:blue")
    ax2 = ax1.twinx()
    ax2_ylabel = ax2.set_ylabel(y2lab)
    ax2_ylabel.set_color('tab:orange')
    ax2_ylabel.set_size(14)
    ax2.set_ylim(xmin_y2-y2_range, xmax_y2+y2_range)
    ax2.scatter(d[x], 
                d[y2], 
                color="tab:orange")
    ax1_xlabel = ax1.set_xlabel(xlab)
    ax1_xlabel.set_size(14)
    ax1.set_xticks([x+1 for x in range(len(d))])

    if idx_list: 
        for idx in idx_list: 
            rect = patches.Rectangle((idx-0.5, 0), 1, 1, linewidth=1, 
                                    edgecolor='r', facecolor='r',
                                    alpha = 0.2)
            ax2.add_patch(rect)
            q = d.iloc[idx,:]['question']
            y = d.iloc[idx,:]['probability']
            ax2.annotate(q, (idx, y), xytext=(idx-0.2, 0.1), rotation=90)

    if outname: 
        plt.savefig(f'../fig/param_surface_config/{outname}.pdf', dpi=300, bbox_inches='tight')
    else: 
        plt.show();

# read configuration-level data
macro_questions = pd.read_csv('../data/analysis/macro_questions.csv')
macro_diagonal = pd.read_csv('../data/analysis/macro_diagonal.csv')

# read parameter-level data
hi_params = pd.read_csv('../data/analysis/hi_params.csv')
Jij_params = pd.read_csv('../data/analysis/Jij_params.csv')

# read surface-level 
raw_means = pd.read_csv('../data/analysis/raw_means.csv')
raw_corr = pd.read_csv('../data/analysis/raw_correlations.csv')

# read question reference 
question_reference = pd.read_csv('../data/analysis/question_reference.csv')
question_reference = question_reference[['question_id', 'question']]

# merge together the three levels of analysis + add question reference
question_level = hi_params.merge(macro_questions, on='question_id', how='inner')
question_level = question_level.merge(raw_means, on='question_id', how='inner')
question_level = question_level.merge(question_reference, on='question_id', how='inner')

coupling_level = Jij_params.merge(macro_diagonal, on=['q1', 'q2'], how='inner')
coupling_level = coupling_level.merge(raw_corr, on=['q1', 'q2'], how='inner')
question_reference_left = question_reference.rename(columns={'question_id': 'q1', 'question': 'question1'})
question_reference_right = question_reference.rename(columns={'question_id': 'q2', 'question': 'question2'})
coupling_level = coupling_level.merge(question_reference_left, on='q1', how='inner')
coupling_level = coupling_level.merge(question_reference_right, on='q2', how='inner')

# look at hi ordering 
question_level_sorted = question_level.sort_values(by=['h'], ascending=False).reset_index(drop=True)
question_level_sorted['idx'] = question_level_sorted.index+1
compare_hi(d = question_level_sorted,
           x = 'idx', 
           xlab = 'question (sorted by local field)', 
           y1 = 'h', 
           y1lab = 'local field', 
           y2 = 'probability', 
           y2lab = 'probability mass', 
           idx_list = [3, 5, 7, 8, 11, 14],
           outname = 'hi_pmass_sortby_hi')

question_level_sorted = question_level.sort_values(by=['probability'], ascending=False).reset_index(drop=True)
question_level_sorted['idx'] = question_level_sorted.index+1
compare_hi(d = question_level_sorted,
           x = 'idx',
           xlab = 'question (sorted by probability mass)',
           y1 = 'h',
           y1lab = 'local field',
           y2 = 'probability',
           y2lab = 'probability mass',
           idx_list = [3, 5, 7, 8, 11, 14],
           outname = 'hi_pmass_sortby_pmass')

compare_hi(d = question_level_sorted,
           x = 'idx',
           xlab = 'question (sorted by probability mass)',
           y1 = 'mean',
           y1lab = 'mean (full records)',
           y2 = 'probability',
           y2lab = 'probability mass',
           idx_list = [3, 5, 7, 8, 11, 14],
           outname = 'mean_pmass_sortby_pmass')
           
# look at Jij ordering
coupling_level['q1_q2'] = coupling_level['q1'].astype(str) + '_' + coupling_level['q2'].astype(str)
coupling_level_sorted = coupling_level.sort_values(by=['Jij'], ascending=False).reset_index(drop=True)
coupling_level_sorted['idx'] = coupling_level_sorted.index+1

compare_hi(d = coupling_level_sorted,
           x = 'idx',
           xlab = 'coupling (sorted by coupling strength)',
           y1 = 'Jij',
           y1lab = 'coupling strength',
           y2 = 'probability',
           y2lab = 'probability mass',
           outname = 'Jij_pmass_sortby_Jij')

coupling_level_sorted = coupling_level.sort_values(by=['probability'], ascending=False).reset_index(drop=True)
coupling_level_sorted['idx'] = coupling_level_sorted.index+1
compare_hi(d = coupling_level_sorted,
           x = 'idx',
           xlab = 'coupling (sorted by probability mass)',
           y1 = 'Jij',
           y1lab = 'coupling strength',
           y2 = 'probability',
           y2lab = 'probability mass',
           outname = 'Jij_pmass_sortby_pmass')

compare_hi(d = coupling_level_sorted,
           x = 'idx',
           xlab = 'coupling (sorted by probability mass)',
           y1 = 'correlation',
           y1lab = 'correlation (full records)',
           y2 = 'probability',
           y2lab = 'probability mass',
           outname = 'correlation_pmass_sortby_pmass')