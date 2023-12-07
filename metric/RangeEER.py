#!/usr/bin/env python

# Copyright 2023 National Institute of Informatics (author: Lin Zhang, zhanglin@nii.ac.jp)
# Licensed under the BSD 3-Clause License.

"""
Calculating Range-based EER for spoof localization. 

Lin Zhang, Xin Wang, Erica Cooper, Nicholas Evans, Junichi Yamagishi (2023) 
Range-Based Equal Error Rate for Spoof Localization. Proc. INTERSPEECH 2023, 3212-3216, 
https://www.isca-speech.org/archive/interspeech_2023/zhang23v_interspeech.html

"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import pyannote.metrics.detection 
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm

from sandbox import eval_asvspoof
from collections import defaultdict
from scipy.stats import percentileofscore

import rle


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--hyp_sco_ali',type=str, 
    help="Path to the predicted score alignment pkl (*_score_ali_*) file.")
parser.add_argument('--hyp_rttm',type=str, 
    help="Path to the predicted score file in RTTM format. Use this or the sco_ali file for performance measurement.")
parser.add_argument('--ref_rttm',type=str, default='../../../database/dev/con_data/rttm_2cls_0sil',
    help="Path to the ground-truth timestamp annotations file in the rttm format.")
parser.add_argument('--save_dir',type=str, default='Loc_RangeEER/singlereso_64',
    help="Directory for saving calculated results.")
parser.add_argument('-u', '--uem',type=str, default='', help="Path to the Un-partitioned Evaluation Map (UEM) file.")
parser.add_argument('--reco2dur_file',type=str, default='../../../database/dev/con_data/reco2dur',
    help="Path to the recording duration file.")
parser.add_argument('--scale',type=float, default=2, help="Resolution of the predicted scores.")

# Binary search algorithm for calculating range-based EER
parser.add_argument('--th_left_per',type=float, default=0, help = "Lower percentile for threshold quantile.")
parser.add_argument('--th_right_per',type=float, default=100, help= "Upper percentile for threshold quantile.")
parser.add_argument('--th_left',type=float, help="Lower threshold in value.")
parser.add_argument('--th_right',type=float, help="Upper threshold in value.")
parser.add_argument('--prec',type=float, default=0.00001, 
    help = """Precision threshold for stopping the search. 
    The process stops when the difference between FNR and FPR is less than this value.""")
parser.add_argument('--metric_base',type=str, default='spoof', choices=['bonafide', 'spoof'])
args = parser.parse_args()


eer_res = []
# threshold fpr fnr abs(fpr-fnr)

if(not os.path.exists(args.save_dir)):
    os.makedirs(args.save_dir)

#0. load groundtruth from rttm
if(args.uem):
    uem_all =np.array([line.split() for line in open(args.uem)])

ref_rttm = load_rttm(args.ref_rttm)
reco2dur = dict([line.split() for line in open(args.reco2dur_file)])
LAB2NUM={'genune':1.0, 'spoof':0.0} #float
def _compute_rate(num, denom):
    if denom == 0.0:
        if num == 0.0:
            return 0.0
        return 1.0
    return num/denom

def cal_eer_by_scolab(sco_lab_pd, return_fpr_fnr=False):
    """
    Normal form is for eer calculation on all data.
    utteer_flag = True for only one utterance's score.
    """
    if ('True_Class' in sco_lab_pd.columns):
        gen=sco_lab_pd[sco_lab_pd.True_Class==LAB2NUM['genune']].Predict_Score_1 #bonafide score is Score_1
        spf=sco_lab_pd[sco_lab_pd.True_Class==LAB2NUM['spoof']].Predict_Score_1
    elif ('True_SpoofRatio' in sco_lab_pd.columns):
        sco_lab_pd['True_SpoofRatio']=sco_lab_pd['True_SpoofRatio'].astype(float)
        gen=sco_lab_pd[sco_lab_pd.True_SpoofRatio==0.0].Predict_Score_1
        spf=sco_lab_pd[sco_lab_pd.True_SpoofRatio>0.0].Predict_Score_1

    fpr, fnr, thresholds = eval_asvspoof.compute_det_curve(gen, spf)

    abs_diffs = np.abs(frr - fpr)
    min_index = np.argmin(abs_diffs)

    if(return_fpr_fnr):
        return frr[min_index], fpr[min_index], thresholds[min_index]
    else:
        eer = np.mean((frr[min_index], fpr[min_index]))
        return eer, thresholds[min_index]


def get_rttm(rttm_file):
    rttm = defaultdict(list)
    with open(rttm_file) as f:
        for line in f.readlines():
            _, reco , channels, st, dur, _, _, lab, _, _  = line.split() 
            rttm[reco].append([float(st), float(st) + float(dur), lab ])
    return rttm

def load_scolab(file_path, pred_cls = True):
    print("loading "+file_path)
    sco_lab = np.load(file_path,allow_pickle=True)
   
    if(sco_lab[0].shape[1]==2):
        #sco_lab_pd = pd.DataFrame(sco_lab,
        #                       columns=('Predict_Score_0', 'Predict_Score_1'))
        #sco_lab_pd = sco_lab_pd.astype({"Predict_Score_0":float, "Predict_Score_1":float})

        sco_lab_pd = pd.DataFrame(columns=('Uttid', 'Predict_Score_0', 'Predict_Score_1'))
        if("score_uttseg" in file_path):
            filenames=np.load(file_path.replace("score_uttseg","filename"),allow_pickle=True)
        elif("score_segseg" in file_path):
            filenames=np.load(file_path.replace("score_segseg","filename"),allow_pickle=True)
        elif("_score_2021" in file_path):
            filenames=np.load(file_path.replace("_score_","_filename_"),allow_pickle=True)

        for name, sco in zip(filenames, sco_lab):
            t = np.concatenate((np.array(name * sco.shape[0]).reshape(-1,1), sco),axis=1)
            sco_lab_pd = pd.concat([sco_lab_pd, pd.DataFrame(t, columns=('Uttid', 'Predict_Score_0', 'Predict_Score_1'))], axis=0, ignore_index=True)

        if("-LCNN-LSTM" in file_path):
            for r, d, f in os.walk(os.path.dirname(file_path)):
                for fname in f:
                    if("2mod_score" in fname):
                        mod2_path = os.path.join(r, fname)
                        print("loading "+mod2_path)
                        sco_lab_2mod = np.load(mod2_path,allow_pickle=True)
                        filenames_2mod =np.load(mod2_path.replace("_score_","_filename_"),allow_pickle=True)

                        for name,sco in zip(filenames_2mod, sco_lab_2mod):
                            t = np.concatenate((np.array(name * sco.shape[0]).reshape(-1,1), sco),axis=1)
                            sco_lab_pd = pd.concat([sco_lab_pd, pd.DataFrame(t, columns=('Uttid', 'Predict_Score_0', 'Predict_Score_1'))], axis=0, ignore_index=True)

        sco_lab_pd = sco_lab_pd.astype({"Uttid": str, "Predict_Score_0":float, "Predict_Score_1":float})
    elif(sco_lab[0].shape[1]==3):
        sco_lab = np.concatenate(sco_lab)
        sco_lab_pd = pd.DataFrame(sco_lab, 
                               columns=('Uttid', 'Predict_Score_0', 'Predict_Score_1'))
        sco_lab_pd = sco_lab_pd.astype({"Uttid": str, "Predict_Score_0":float, "Predict_Score_1":float})
    else:
        sco_lab = np.concatenate(sco_lab)
        sco_lab_pd = pd.DataFrame(sco_lab, 
                               columns=('Uttid', 'Predict_Score_0', 'Predict_Score_1','True_Class' ))
        sco_lab_pd = sco_lab_pd.astype({"Uttid": str, "True_Class": float, "Predict_Score_0":float, "Predict_Score_1":float})
        #sys.getsizeof(float())=24
        #sco_lab_pd = sco_lab_pd.astype({"True_Class": np.uint8})

    if(pred_cls):

        eer, bona_th = cal_eer_by_scolab(sco_lab_pd)
        sco_lab_pd['Pred_Class'] = 'spoof'
        lst = sco_lab_pd['Predict_Score_1'] > bona_th
        if(len(lst)>0):
            sco_lab_pd.loc[lst, 'Pred_Class'] = 'bonafide'
        #here we set true_class as float since we will use rolling windows to extract min value 
    return sco_lab_pd.drop(columns=['Predict_Score_0']) #save space.


def get_pd_class_by_th(sco_pd, th):

    sco_pd['Pred_Class'] = 'spoof'
    lst = sco_pd['Predict_Score_1'] > th
    if(len(lst)>0):
        sco_pd.loc[lst, 'Pred_Class'] = 'bonafide'

    return sco_pd



def count_continous(vadvec):
    return rle.encode(vadvec)


def scoali2seg(scoali, scale ):
    """
    scale in ms
    """
    scale_s = float(scale / 1000.0)
    res = Annotation()
    #1. convert to vad
    vadvec = (scoali['Pred_Class']==args.metric_base)
    lab_vec, count_vec = count_continous(vadvec.tolist())
    st = 0
    for lab, c in zip(lab_vec, count_vec):
        et = st + float(c) * scale_s
        if(not lab):
            st = et
            continue
        res[Segment(st, et)] = args.metric_base
        st = et

    return res

def get_detcos_by_th(sco_pd, th):
    detCos = pyannote.metrics.detection.DetectionCostFunction()

    sco_lab_pd = get_pd_class_by_th(sco_pd, th)

    for uttid in np.unique(sco_pd['Uttid']):
        if(uttid not in ref_rttm):
            ref = Annotation()  #bonafide utterance
            ##debug 0215, didn't count bonafide utterance.
            if(uttid.startswith('CON')):
                print("WARNING: didn't count error for {}".format(uttid))
                continue
        else:
            ref = ref_rttm[uttid]

        if(args.uem):
            uem_lst = uem_all[np.where(uem_all[:,0] == uttid)] 
            uem = Timeline(segments=[Segment(float(st), float(et)) for uid, st, et in uem_lst],\
                    uri=uttid)
        else:
            uem = Segment(0, float(reco2dur[uttid])) #None

        hyp = scoali2seg(sco_lab_pd[sco_lab_pd['Uttid']==uttid], scale = args.scale*10.0)

        detcos_detail = detCos(reference=ref, hypothesis=hyp , uem=uem, detailed=True) 
    fpr = _compute_rate(detCos.accumulated_['false alarm'], detCos.accumulated_['negative class total'] )
    fnr = _compute_rate(detCos.accumulated_['miss'], detCos.accumulated_['positive class total'] )

    return fpr, fnr

def initial(args, sco_pd):
    sco_all = np.sort(sco_pd.Predict_Score_1.tolist())    
    #based on score
    th_left, th_right = min(sco_all), max(sco_all)

    #based on old redults.
    init_path = os.path.join(args.save_dir,"RangeEER_det")
    if(os.path.exists(init_path)):
        eer_init = np.loadtxt(init_path).reshape(-1, 4)
        eer_init_left = eer_init[eer_init[:, 1] < eer_init[:, 2]]
        eer_init_right = eer_init[eer_init[:, 1] >= eer_init[:, 2]]
        if(len(eer_init_left) > 0):
            th_left = max(eer_init_left[-1, 0], th_left)
        if(len(eer_init_right) > 0):
            th_right = min(eer_init_right[-1, 0], th_right)

    #based on config
    #can be set by percentile or value
    if(args.th_left):
        th_left = max(th_left, args.th_left)
    if(args.th_right):
        th_right = min(th_right, args.th_right)

    th_left = max(th_left, np.percentile(sco_all, args.th_left_per))
    th_right = min(th_right, np.percentile(sco_all, args.th_right_per))

    th_left_per = percentileofscore(sco_all, th_left)
    th_right_per = percentileofscore(sco_all, th_right)
 
    if(th_left_per == 0 and th_right_per == 100):
        #1.1 find percentile for segment eer
        eer, th_mid = cal_eer_by_scolab(sco_pd)
        th_mid_per = percentileofscore(sco_all, th_mid)
    else:
        th_mid_per = (th_left_per + th_right_per)/(2.0)
        th_mid = np.percentile(sco_all, th_mid_per)

    return th_left, th_left_per, th_right, th_right_per, th_mid, th_mid_per

def main():
    eer_res=[] #List of [threshold, fpr, fnr, diff]
    save_file = os.path.join(args.save_dir,"RangeEER_det")

    #2. set class based on score
    #3. scoali2seg : [0,0,1,1,0,...] -> annotation
    #4. pyannote,

    if(args.hyp_rttm):
        raise NotImplementedError
        hyp_rttm = load_rttm(args.hyp_rttm)
        utt_lst = hyp_rttm.keys()
    elif(args.hyp_sco_ali):
        sco_pd  = load_scolab(args.hyp_sco_ali, pred_cls = False)
        #utt_lst = np.unique(sco_pd['Uttid'])

    #1. init
    #1.0 init returned res list.

    #1.2. set quantile for left, mid and right.
    th_left, th_left_per, th_right, th_right_per, th_mid, th_mid_per = initial(args, sco_pd)

    #1.3 get init fpr and fnr from mid 
    fpr_left, fnr_left = get_detcos_by_th(sco_pd, th_left)
    fpr_mid, fnr_mid = get_detcos_by_th(sco_pd, th_mid)
    eer_res.append([th_mid, fpr_mid, fnr_mid, abs(fpr_mid-fnr_mid)])

    #2. bisect
    #early_tag = True #we'd like to start with the point fpr < fnr.
    while(th_left < th_right and abs(fpr_mid-fnr_mid) > args.prec ):

        if((fpr_left - fnr_left) * (fpr_mid - fnr_mid) < 0 ):
            th_right, th_right_per = th_mid, th_mid_per
            fpr_right, fnr_right = fpr_mid, fnr_mid
        else:
            #if((fpr_right - fnr_right) * (fpr_right - fnr_right) < 0 ):
            th_left, th_left_per = th_mid, th_mid_per
            fpr_left, fnr_left = fpr_mid, fnr_mid

        th_mid_per = (th_left_per + th_right_per)/2.0
        th_mid= np.percentile(sco_pd.Predict_Score_1, th_mid_per)
        fpr_mid, fnr_mid = get_detcos_by_th(sco_pd, th_mid)

        eer_res.append([th_mid, fpr_mid, fnr_mid, abs(fpr_mid-fnr_mid)])
        with open(save_file, 'ab') as f:
            np.savetxt( f, np.array([th_mid, fpr_mid, fnr_mid, abs(fpr_mid - fnr_mid)]).reshape(-1,4))
        print(np.array([th_mid, fpr_mid, fnr_mid, abs(fpr_mid-fnr_mid)]).reshape(-1, 4))    

    if(os.path.exists(save_file)):
        eer_ary = np.loadtxt(save_file).reshape(-1, 4)
    eer_res = eer_ary[(-eer_ary[:, -1]).argsort()[:]].tolist()
    min_idx = np.argmin(eer_ary[:,3])

    print("RangeEER={}, with threshold = {}".format(np.mean(eer_res[min_idx][1:2]), th_mid))

if __name__ == '__main__':
    main()



