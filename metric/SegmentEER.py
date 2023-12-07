
#!/usr/bin/env python

# Copyright 2023 National Institute of Informatics (author: Lin Zhang, zhanglin@nii.ac.jp)
# Licensed under the BSD 3-Clause License.

"""
Calculating point-based segment-level EER for spoof localization. 
mean for round with different random seeds
(use spicial to limit the pkl.)

As mentioned in Sec. 4.3 of [Range-Based Equal Error Rate for Spoof Localization]
(https://arxiv.org/pdf/2305.17739.pdf), we can up-sample or down-sample the predicted scores to other reso.
This function is used to downsample/upsample the predicted score as shown in Fig.2 

Upsample: extend this to get finer prediction from coaser one. like upsample 20ms to 40ms
20ms: |--0.3--|--0.8--|
40ms: |0.3|0.3|0.8|0.8|

Downsample: extend this to get finer prediction from coaser one. like upsample 20ms to 40ms
40ms: |0.3|0.6|0.8|0.9|
20ms: |--0.3--|--0.8--|

Requires model_dir and args.save_dir

Usage: $: python SegmentEER.py --model_dir <model_dir> --save_dir <save_dir> --sml_dir . 
"""

import os
import sys
import pickle
import librosa
import argparse
import numpy as np
import pandas as pd
import scipy.io.wavfile as sciwav

from numpy import nan
from tqdm import tqdm
from sandbox import eval_asvspoof
from collections import defaultdict
from rttm_tool import get_rttm, get_vadvec, rttm2vadvec
np.set_printoptions(linewidth=100000)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--model_dir',type=str, default='PartialSpoof/03multireso/multi-reso')
parser.add_argument('--sml_dir_list',type=str, nargs='+', default=['exp-01', 'exp-02', 'exp-03'], 
        help="this is for sml_dir with different random seeds. But you can set this as . if the model_dir is the only experiment dir")
parser.add_argument('--save_dir',type=str, default='Loc_SegmentEER/singlereso/64')
parser.add_argument('--rttm_file',type=str, default='../../../database/dev/con_data/rttm_2cls_0sil')
parser.add_argument('--reco2dur_file',type=str, default='../../../database/dev/con_data/reco2dur')
parser.add_argument('--keyword',type=str, default='2023', 
        help="""specify the keyword in the score_ali pkl, for example, you can set this to evaluate files with specific date.
        or you can set this for score_ali pkl file with specific scoring branch. """)

parser.add_argument('--use_ext_flag', action='store_true', default=False, 
        help="load existing results directly or re-calculate it")
parser.add_argument('--print_each', action='store_true', default=True, 
        help="whether print each EER for all random seeds.") 
parser.add_argument('--print_var',action='store_true', default=False, 
        help="print the variance of EER with different random seeds") 

# set options for the EER matrix. 
parser.add_argument('--DIGEER',action='store_true', default=True, 
        help="Whether calculate EER using the (diagnal of EER matrix) ")
parser.add_argument('--UpEER',action='store_true', default=True, 
        help="Whether downsample the predicted score to coarse-grained reso., (EERs in the upper triangle.)")
parser.add_argument('--LowEER',action='store_true', default=True, 
        help="Whether upsample the predicted score to fine-grained reso., (EERs in the lower triangle.)")
parser.add_argument('--UTTEER',action='store_true', default=True, 
        help="whether calculate EER for utterance-level spoof detection by using segment-level scores. will shown in the last column.")
args = parser.parse_args()


Scale_num=7   
SSL_shift=1   ##since SSL use 20ms as frame shift...
Base_step=0.01 #base in sec
Pred_Frame_shifts= np.array([pow(2, i) for i in np.arange(Scale_num)])[SSL_shift:] #Use which score branches to derive predicted scores? 
Measure_Frame_shifts= np.array([pow(2, i) for i in np.arange(Scale_num)])[:] #Measure perfomrance on which reso.?


rttm= get_rttm(args.rttm_file)
reco2dur=dict([line.split() for line in open(args.reco2dur_file)])
label2num_file = "../../../database/label2num/label2num_2cls_0sil"
LAB2NUM=dict([ [line.split()[0], float(line.split()[1])] for line in open(label2num_file) ])

DSETs=['dev', 'eval']


assert(args.DIGEER + args.UpEER + args.LowEER >0)
if( not os.path.exists(args.save_dir) ):
    os.makedirs(args.save_dir)

def load_scolab(file_path):
    """
    load score_ali pkl, and save it to pandas.dataframe, which is for further upsample/downsample.
    """
    print("loading "+file_path)
    sco_lab = np.load(file_path,allow_pickle=True)
    sco_lab = np.concatenate(sco_lab)

    sco_lab_pd = pd.DataFrame(sco_lab, 
                              columns=('Wavid', 'Predict_Score_0', 'Predict_Score_1','True_Class' ))
    sco_lab_pd = sco_lab_pd.astype({"Wavid": str, "True_Class": float, "Predict_Score_0":float, "Predict_Score_1":float})
    #here we set true_class as float since we will use rolling windows to extract min value 
    return sco_lab_pd

def cal_eer_by_scolab(sco_lab_pd, SpoofRatio=False, utteer_flag = False ):
    """
    Normal form is for eer calculation on all data.
    utteer_flag = True for only one utterance's score.
    """
    if ('True_Class' in sco_lab_pd.columns):
        gen=sco_lab_pd[sco_lab_pd.True_Class==LAB2NUM['bonafide']].Predict_Score_1 #bonafide score is Score_1
        spf=sco_lab_pd[sco_lab_pd.True_Class==LAB2NUM['spoof']].Predict_Score_1
    elif ('True_SpoofRatio' in sco_lab_pd.columns):
        sco_lab_pd['True_SpoofRatio']=sco_lab_pd['True_SpoofRatio'].astype(float)
        gen=sco_lab_pd[sco_lab_pd.True_SpoofRatio==0.0].Predict_Score_1
        spf=sco_lab_pd[sco_lab_pd.True_SpoofRatio>0.0].Predict_Score_1

    eer, threshold = eval_asvspoof.compute_eer(gen, spf)
    if(utteer_flag):
        return eer

    return eer, threshold

def get_pd_in_scoretype(sco_lab_pd, fs, SCORE_TYPE='min', return_eer=False ):
    """
    This function is used to downsample the predicted score, 
    using SCORE_TYPE to choose score, and with the sliding wiindow (frame_shift) as fs
    """
    fs = min(len(sco_lab_pd), fs) # for the case when fs > len(pd).
    if(SCORE_TYPE == 'mean'):
        lab = sco_lab_pd['True_Class'].rolling(window=fs).min()[fs-1::fs]
        score = sco_lab_pd['Predict_Score_1'].rolling(window=fs).mean()[fs-1::fs]
        new_pd = pd.concat([lab, score],axis =1)
    elif(SCORE_TYPE == 'max'):
        lab = sco_lab_pd['True_Class'].rolling(window=fs).min()[fs-1::fs]
        score = sco_lab_pd['Predict_Score_1'].rolling(window=fs).max()[fs-1::fs]
        new_pd = pd.concat([lab, score],axis =1)
    else: #default using min
        new_pd = sco_lab_pd[['Predict_Score_1', 'True_Class']].rolling(window=fs).min()[fs-1::fs]

    if(len(new_pd)< len(sco_lab_pd)/fs):
        #for the rest part, longer than shift windows.
        t = sco_lab_pd[['Predict_Score_1', 'True_Class']][int(fs*(len(sco_lab_pd)//fs)):].min()
        #new_pd = new_pd.append(t, ignore_index=True)
        new_pd = pd.concat([new_pd, t.set_axis(new_pd.columns).to_frame().T], ignore_index=True)

    return new_pd    

def get_dupary_by_scale(sco_lab_pd, scale, to_scale, scale_change_time=0):
    """
    To upsample the predicted score to a finer-grained reso, we need to duplicate them
    """
    uttid = np.unique(sco_lab_pd['Wavid'])[0]
    #1. duplicate for score
    scale_change_time = scale_change_time or (scale / to_scale)
    new_pd = pd.DataFrame(np.repeat(sco_lab_pd.to_numpy(), scale_change_time, axis=0),
            columns=sco_lab_pd.columns)

    #2. get new fround-truth based on the new scale.
    gt_lab = rttm2vadvec(rttm[uttid], to_scale * Base_step, reco2dur[uttid], skip_last=False, 
        label2num_file=label2num_file)
    lab_num = min(len(new_pd), len(gt_lab))
    del_num = max(len(new_pd), len(gt_lab)) - lab_num
    pad = [nan for i in np.arange(del_num)] if (len(new_pd)>len(gt_lab)) else []

    new_pd['True_Class'] = np.hstack((gt_lab[:lab_num], pad))
    new_pd=new_pd.dropna().reset_index(drop=True)

    return new_pd['Predict_Score_1'], new_pd['True_Class']    

def get_duppd_by_scale(sco_lab_pd, scale, to_scale, scale_change_time=0):
    uttid = np.unique(sco_lab_pd['Wavid'])[0]
    #1. duplicate for score
    scale_change_time = scale_change_time or (scale / to_scale)
    new_pd = pd.DataFrame(np.repeat(sco_lab_pd.to_numpy(), scale_change_time, axis=0),
            columns=sco_lab_pd.columns)

    #2. get new fround-truth based on the new scale.
    gt_lab = rttm2vadvec(rttm[uttid], to_scale * Base_step, reco2dur[uttid], skip_last=False, 
        label2num_file=label2num_file)
    lab_num = min(len(new_pd), len(gt_lab))
    del_num = max(len(new_pd), len(gt_lab)) - lab_num
    pad = [nan for i in np.arange(del_num)] if (len(new_pd)>len(gt_lab)) else []

    new_pd['True_Class'] = np.hstack((gt_lab[:lab_num], pad))
    new_pd=new_pd.dropna().reset_index(drop=True)

    return new_pd    

def get_eermatrix_scale(sco_lab_pd, scale):
    """
    This function is used to derive the EER matrix, 
    EERs in the diagonal represent the EER for the corresponding resolution.
    EERs in the upper triangle
    And note that upper triangle may be an “underestimation”, while lower triangle is more correct. 
    """

    eers=np.zeros(len(Measure_Frame_shifts)) #* np.inf
    thresholds=np.zeros(len(Measure_Frame_shifts)) #* np.inf

    index = Measure_Frame_shifts.tolist().index(scale)
    measure_scale = np.arange(index, len(Measure_Frame_shifts))
    if(args.LowEER):
        measure_scale = np.arange(Scale_num)
        
    new_pd = sco_lab_pd
    for idx in measure_scale:
        if(idx==index and args.DIGEER):   
            new_pd = sco_lab_pd.copy()
            if(args.UTTEER):
                utteer_pd = sco_lab_pd.groupby('Wavid').apply(cal_eer_by_scolab, utteer_flag=True)
                utteer_pd = pd.DataFrame(utteer_pd).reset_index()
                utteer_pd.rename( columns={0:'EER'}, inplace=True  )
        elif(idx > index and args.UpEER): 
            #if we want to calculate eer in the upper diag.
            fs = int(Measure_Frame_shifts[idx] / scale) 
            #we need to divide scale, since the further eer calculation is based on frame shift on basic scale.
            # any function passed to apply is applied along each column/series independent

            # we only use Predict_score_1, so we only need to consider one column:
            #  and since here we use Cosine similarity, so higher means more similar, Score_1 for similar to bonafide. 
            #  -> smaller score_1 means spoof.
            new_pd = sco_lab_pd.groupby('Wavid').apply(get_pd_in_scoretype, fs = fs)  

        elif(idx < index and args.LowEER):
            new_pd = sco_lab_pd.groupby('Wavid').apply(get_duppd_by_scale, scale, to_scale = Measure_Frame_shifts[idx]) 
        else:
            continue

        eers[idx], thresholds[idx] = cal_eer_by_scolab(new_pd)    

    if(args.UTTEER):
        return eers, thresholds, utteer_pd    
    return eers, thresholds, ''    

def get_spsco_eer_from_pd(sco_lab_pd, ori_scale):
    gen_all, spf_all = [], []
    for uttid in np.unique(sco_lab_pd.Wavid):
        pred_score, true_class = get_dupary_by_scale(sco_lab_pd[sco_lab_pd.Wavid == uttid], ori_scale, to_scale = 1/160.0)
        gen = np.array(pred_score[true_class == LAB2NUM['bonafide']].astype('float64')) 
        spf = np.array(pred_score[true_class == LAB2NUM['spoof']].astype('float64'))
        gen_all.append(gen)
        spf_all.append(spf)

    return np.concatenate(gen_all), np.concatenate(spf_all)


def get_utteer_by_seg(sco_lab_pd):
    """
    Measure Utterance EER based on segment-level score vectors.
    For each utteracne, using min() to choose the minimum segment score as the utterance score. 
    """
    eer_utt, threshold_utt = cal_eer_by_scolab(sco_lab_pd.groupby(['Wavid']).min())

    return eer_utt, threshold_utt


def get_mean_multi_seed(eer_scales_seed, threshold_scales_seed, return_var=False):
    """
    Calculate average of EERs with different initial random seeds.
    """
    if(return_var):
        return np.mean(eer_scales_seed,axis = 0), np.mean(threshold_scales_seed,axis=0), \
               np.var(eer_scales_seed,axis = 0), np.var(threshold_scales_seed,axis=0)

    return np.mean(eer_scales_seed,axis = 0), np.mean(threshold_scales_seed,axis=0), None, None

#3. eer


def print_res(res_data):
    """
    print results
    """
    eers_scales_seed_mean, threshold_scales_seed_mean, eers_scales_seed_var, _= get_mean_multi_seed(
            res_data['eers_scales_seed'], res_data['threshold_scales_seed'], return_var=args.print_var)
    
    if(args.print_each):
        print('-'*20+'eer for each res')
        print(res_data['eers_scales_seed']*100)
        print('-'*20+'threshold for each res')
        print(res_data['threshold_scales_seed'])

    print('-'*20+'mean')
    print(eers_scales_seed_mean * 100)
    if(args.print_var):
        print('-'*20+'var')
        print(eers_scales_seed_var * 100)


def main():
    np.set_printoptions(linewidth=100000)

    eers_scales_seed = []
    threshold_scales_seed = []
    utteer_scales_seed = []

    for dset in DSETs:
        eer_file='{}/SegmentEER_{}'.format(args.save_dir, dset)

        if(os.path.exists(eer_file+'.npz') and (args.use_ext_flag)):
            # load pre-calculated EER matrix
            res_data = np.load(eer_file+".npz", allow_pickle=True)

            print("="*10,"{} with {} seeds".format(
                dset, len(res_data['eers_scales_seed'])),"="*10)
            print_res(res_data)

        else:
            for sml_name in args.sml_dir_list: 
                #read all models (especially with different seeds) within this dir.
                sml_dir = os.path.join(args.model_dir, sml_name)   

                eer_scales=[]
                threshold_scales=[]
                utteer_scales = {}

                model_out_dir=os.path.join(args.model_dir, sml_dir,'output')
                if(os.path.exists(model_out_dir)):
                    #to judge wether we have aligned score .pkl file.
                    #and this is not temperate folder.
                    find_it=False
                    while(True):
                        #to find the time_tag based on the ali.npz.
                        for score_name in os.listdir(model_out_dir):
                            name, app = os.path.splitext(score_name)
                            if ( name.find(args.keyword)>=0 and name.find(f'{dset}_score_ali')>=0  and app == '.pkl'):
                                find_it=True; break
                        if(find_it): break;
                        print("Can not find score_ali file, please check " + model_out_dir)
                        sys.exit()
                    time_tag=name.split('_')[-1]

                    for scale in np.flip(Pred_Frame_shifts):
                    #Iterate all score branches listed in Pred_Frame_shifts
                        sco_ali_file=os.path.join(model_out_dir,
                                '{}_score_ali_{}_{}.pkl'.format(dset, scale,time_tag))
                        sco_lab_pd = load_scolab(sco_ali_file)

                        #segment eer vector. Evaluate this score branch in all possible reso. eers's dim: [1, len(Measure_Frame_shifts)]
                        eers, thresholds, utteer_pd = get_eermatrix_scale( sco_lab_pd, scale) 

                        #utterance eer from segments score
                        eer_utt, threshold_utt = get_utteer_by_seg(sco_lab_pd)

                        #append eer(segment eer in x Scale_num; utterance eer)
                        eer_scales.append(np.concatenate((eers, [eer_utt])))
                        threshold_scales.append(np.concatenate((thresholds, [threshold_utt])))
                        utteer_scales[f'{scale}']=utteer_pd

                    # eer_scales[0].shape[0] or len(Measure_Frame_shifts) + 1, which represents dim as reso. + utterance (1) 
                    eers_scales_seed.append(np.concatenate(eer_scales).reshape(-1, eer_scales[0].shape[0]))   
                    threshold_scales_seed.append(np.concatenate(threshold_scales).reshape(-1, eer_scales[0].shape[0]))
                    utteer_scales_seed.append(utteer_scales)

            np.savez(eer_file, 
                    eers_scales_seed = eers_scales_seed, threshold_scales_seed = threshold_scales_seed,
                    utteer_scales_seed = utteer_scales_seed)
            #final column is utterance eer from segments score.

            eers_scales_seed_mean, threshold_scales_seed_mean, _, _ = get_mean_multi_seed(
                    eers_scales_seed, threshold_scales_seed)
            print("="*10,"{} {}".format(args.model_out_dir, dset),"="*10)
            print("eer:\n", eers_scales_seed_mean * 100)
            #print("threshold:\n", threshold_scales_seed_mean)
            
    
if __name__ == "__main__": 
    main()

