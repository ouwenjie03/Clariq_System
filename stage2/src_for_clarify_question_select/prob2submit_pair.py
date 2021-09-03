#!/usr/bin/env python
# encoding: utf-8

"""
@author: ouwenjie
@license: NeteaseGame Licence 
@contact: ouwenjie@corp.netease.com


"""


import sys
import numpy as np
import pandas as pd

model_name = 'NTES_ALONG'

result_names = [
    'pair_devset_for_test_all_electra_large_add_pseudo_result'
]

full_df = None
for rname in result_names:
    print(rname)
    df = pd.read_csv('./{}.tsv'.format(rname), sep='\t')
    if full_df is None:
        full_df = df.copy()
    else:
        full_df['probs'] += df['probs']


fout = open(sys.argv[1], 'w', encoding='utf-8')
for tid in full_df['tid'].unique():
    qids = full_df[full_df['tid'] == tid]['qid'].values
    probs = full_df[full_df['tid'] == tid]['probs'].values
    sorted_idxs = np.argsort(probs)[::-1][:30]
    for i in range(len(sorted_idxs)):
        si = sorted_idxs[i]
        fout.write('{} 0 {} {} {} {}\n'.format(tid, qids[si], i, probs[si], model_name))
fout.close()

