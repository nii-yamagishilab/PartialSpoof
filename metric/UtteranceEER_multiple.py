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
import numpy as np
import pandas as pd
from sandbox import eval_asvspoof

np.set_printoptions(linewidth=100000)


#######Configuration.
PS_PATH = "/home/smg/zhanglin/workspace/PROJ/Public/CODE/PartialSpoof"
DSETs=['dev', 'eval']
print_mean = True #whetehr print mean of EERs for all random seeds 
print_each = True #whetehr print each EER for all random seeds 
SEEDs = [1, 2, 3]
SCORE_TYPE='min'
output_dir='output' 
#output_dir='spf19' 'spf21'
MODEL_NAME="03multireso/single-reso/64"
#MODEL_NAME="03multireso/multi-reso"
#######Configuration.



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
    main_dir = os.path.join(PS_PATH, MODEL_NAME)
    print(main_dir)

    for dset in DSETs:

        mintDCF_seeds = np.zeros(MAX_col-3)
        eer_seeds = np.zeros(MAX_col-3)
        threshold_seeds =np.zeros(MAX_col-3)

        #prepare ASV score
        ASV_SCORES_FILE="{}/database/protocols/PartialSpoof_LA_asv_scores/PartialSpoof.LA.asv.{}.gi.trl.scores.txt".format(PS_PATH, dset)
        # ZhangLin: But I personally do not recommend using t-DCF, because Partial Spoof is designed 
        # not only for ASVspoof, which aims to deceive machines, 
        # but also for DeepFake, which is intended to fool humans.


        trials=pd.read_csv(ASV_SCORES_FILE,error_bad_lines=False,
                sep='\s+',index_col =False, header=None,
                names=['type', 'label', 'score'])
        tar_asv = np.array(trials.loc[trials['label']=='target']['score'], dtype=float)
        non_asv = np.array(trials.loc[(trials['label']=='nontarget')&(trials['type']=='bonafide')]['score'], dtype=float)
        spoof_asv_pd=trials.loc[(trials['label']=='nontarget')&(trials['type']=='spoof')]
        spoof_asv = np.array(spoof_asv_pd['score'], dtype=float)

        print("="*15, "%s  %s with %s seed " % (MODEL_NAME, dset, len(SEEDs)), "="*15) 
        utteers_scales_seed=[]
        uttthreshold_scales_seed=[]
        for seed_idx in SEEDs:
            data_path = os.path.join(main_dir, f'exp-0{seed_idx}', output_dir, f'log_output_{dset}' );
            print(data_path)
            mintDCF_oneseed_cols = []
            eer_oneseed_cols = []
            threshold_oneseed_cols =[]

            for col in np.arange(3, MAX_col):
                bonafide, spoofed = parse_txt(data_path, col)
                
                mintDCF, eer, threshold = eval_asvspoof.tDCF_wrapper(bonafide, spoofed, tar_asv, non_asv, spoof_asv)

                mintDCF_oneseed_cols.append(mintDCF)
                eer_oneseed_cols.append(eer)
                threshold_oneseed_cols.append(threshold)

            if (print_each):
                print('In Seed ' + str(seed_idx))
                print(np.array(eer_oneseed_cols) * 100)
                print('---threshold---')
                print(np.array(threshold_oneseed_cols))

                
            mintDCF_seeds += np.array(mintDCF_oneseed_cols)
            eer_seeds += np.array(eer_oneseed_cols)    
            threshold_seeds += np.array(threshold_oneseed_cols)

            utteers_scales_seed.append(np.array(eer_oneseed_cols))
            uttthreshold_scales_seed.append(np.array(threshold_oneseed_cols))

        if (print_mean):
            eer_seeds_mean = eer_seeds / len(SEEDs)
            mintDCF_mean = mintDCF_seeds / len(SEEDs)
            threshold_mean = threshold_seeds / len(SEEDs)
            
            print('-' * 50)
            print('EER: {}\n'.format(eer_seeds_mean*100),
                  'mintDCF: {}\n'.format(mintDCF_mean),
                  'threshold: {}'.format(threshold_mean))
