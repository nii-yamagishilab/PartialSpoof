import os
import sys
import numpy as np
import argparse

from collections import defaultdict
"""
zlin
"""


#Argument Parser
#   label2num_file="data/label/label2num_all"
default_label2num="/home/smg/zhanglin/workspace/PROJ/01spf-con/00data-prepare/exp-2d-sametol-label/label2num_all"



###############

def get_label2num(label2num_file):
    """
    Get the label2num based on the file. 

    Input: label2num file.

    Output: {label: num} 
        nonspeech: 0 
        A01: 1
    """

    label2num = dict( [line.split() for line in open(label2num_file) ] )
    num2label = {k:v for v,k in label2num.items()}

    return label2num, num2label



def get_rttm(rttm_file):
    rttm = defaultdict(list)
    with open(rttm_file) as f:
        for line in f.readlines():
            _, reco , channels, st, dur, _, _, lab, _, _  = line.split() 
            rttm[reco].append([lab, float(st), float(format(float(st) + float(dur), '.5f')) ])
    return rttm


def get_vadvec(vad_file, seg_shift_sec):
    vad_matrix = np.loadtxt(vad_file)
    n_frames = int(max(vad_matrix[:,1]) / seg_shift_sec)
    vadvec = np.zeros(n_frames)
    for st, et, lab in vad_matrix:
        sf = int(round(st / seg_shift_sec))
        ef = int(round(et / seg_shift_sec))
        if (sf<ef):
            vadvec[sf : ef] = int(lab)
    return vadvec

def rttm2vadvec(rttm, seg_shift_sec, dur=0, skip_last=True, 
        label2num_file=default_label2num):

    """
    convert rttm to vad vector.
    """
    label2num, num2label = get_label2num(label2num_file)
    rttm = np.array(rttm)
    dur = float(dur)
    if(dur>0):
        pass
    else:
        dur = max(rttm[:,2])

    if(skip_last):
        n_frames = int(float(dur) / seg_shift_sec)
    else:
        n_frames = int(np.ceil(float(dur) / seg_shift_sec))
    vadvec = np.zeros(n_frames).astype(int)
    for lab, st, et in rttm:
        sf = int(round(float(st) / seg_shift_sec))
        ef = int(round(float(format(float(et),'.5f')) / seg_shift_sec)) 
        #TODO clean rttm, recover this line.
        #ef = int(round(float(et) / seg_shift_sec))  # rttm has infinite decimal because of PC computing. so I added format here.
        if (sf<ef):
            vadvec[sf : ef] = label2num[lab]
    return vadvec


def savepred_as_seg(args, res):
    #initial
    reco2dur = dict( [line.split() for line in open(args.reco2dur) ])

    seg_shift_sec = args.scale * 0.01
    seg_len_sec = seg_shift_sec

    f_segment = open(args.out_dir+'/res.segment', 'wt') #<seg> <reco> <st> <end>
    f_seglab = open(args.out_dir+'/res.seglab', 'wt')  #<seg> <lab>
    #finish initial
        
    if not isinstance(res, dict):
        res = res.tolist()

    for reco, pred in res.items():
        
        dur = float(reco2dur[reco])
        for idx, p in enumerate(pred):
            st =  idx * seg_shift_sec 
            end = st + seg_len_sec
            if(idx == len(pred)-1):
                end = dur
            segid = str(reco) + '_' + str(idx)
           
            f_segment.write('{} {} {:.4f} {:.4f}\n'.format(segid, reco, st, end))
            
            f_seglab.write('{} {}\n'.format(segid, str(int(p))))
        
    f_segment.close()
    f_seglab.close()
    
