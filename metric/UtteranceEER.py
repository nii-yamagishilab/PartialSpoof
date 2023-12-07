#!/usr/bin/python

"""
Used to evaluate segment model in utterance-level detection
input: output/log_output_dev
       uttscore are in seg1.min, seg2.min, seg3.min, seg4.min,...,utt
usage: 
print: Utterance-level eer for each scales.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sandbox import eval_asvspoof
np.set_printoptions(linewidth=100000)



parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--pred_file',type=str, default='')
parser.add_argument('--asv_score_file',type=str, default='To calculate t-DCF.')
        #ASV_SCORES_FILE="{}/database/protocols/PartialSpoof_LA_asv_scores/PartialSpoof.LA.asv.{}.gi.trl.scores.txt".format(PS_PATH, dset)
        # ZhangLin: But I personally do not recommend using t-DCF, because Partial Spoof is designed 
        # not only for ASVspoof, which aims to deceive machines, 
        # but also for DeepFake, which is intended to fool humans.
args = parser.parse_args()


#######Configuration.
PS_PATH = "/home/smg/zhanglin/workspace/PROJ/Public/CODE/PartialSpoof"
print_mean = True #whetehr print mean of EERs for all random seeds 
print_each = True #whetehr print each EER for all random seeds 
SCORE_TYPE='min'


scale_num = 7
MAX_col = scale_num + 3  #9 3+6
def parse_txt(file_path, col):
    """
    create score lists for bona fide and spoof.
    """
    bonafide = []
    spoofed = []
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            if line.startswith('Output,'):
                temp = line.rstrip('\n').split(',')
                flag = int(temp[2])
                if flag:
                    bonafide.append(float(temp[col]))
                else:
                    #spoofed.append(pow(float(temp[col]),2))
                    spoofed.append(float(temp[col]))
    bonafide = np.array(bonafide)
    spoofed = np.array(spoofed)
    return bonafide, spoofed



if __name__ == "__main__":

    trials=pd.read_csv(args.asv_score_file, on_bad_lines='skip',
            sep='\s+',index_col =False, header=None,
            names=['type', 'label', 'score'])
    tar_asv = np.array(trials.loc[trials['label']=='target']['score'], dtype=float)
    non_asv = np.array(trials.loc[(trials['label']=='nontarget')&(trials['type']=='bonafide')]['score'], dtype=float)
    spoof_asv_pd=trials.loc[(trials['label']=='nontarget')&(trials['type']=='spoof')]
    spoof_asv = np.array(spoof_asv_pd['score'], dtype=float)

    print(args.pred_file)
    mintDCF_oneseed_cols = []
    eer_oneseed_cols = []
    threshold_oneseed_cols =[]

    for col in np.arange(3, MAX_col):
        bonafide, spoofed = parse_txt(args.pred_file, col)
        
        mintDCF, eer, threshold = eval_asvspoof.tDCF_wrapper(bonafide, spoofed, tar_asv, non_asv, spoof_asv)

        mintDCF_oneseed_cols.append(mintDCF)
        eer_oneseed_cols.append(eer)
        threshold_oneseed_cols.append(threshold)

    if (print_each):
        print('===' + str(args.pred_file) + '===')
        print(np.array(eer_oneseed_cols) * 100)
        print('---threshold---')
        print(np.array(threshold_oneseed_cols))
                
