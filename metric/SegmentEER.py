import os
import sys
import librosa
import pickle
import numpy as np
import pandas as pd
import scipy.io.wavfile as sciwav
#import matplotlib.pyplot as plt
#import IPython.display as ipd
from tqdm import tqdm

from collections import defaultdict

from sandbox import eval_asvspoof
np.set_printoptions(linewidth=100000)


"""
EER for segmental prediction
mean for six times round with different rdom seed
(use spicial to limit the pkl.)
This is simple version, which is not consider the extend scales.
"""

Scale_num=7   
SSL_shift=1   ##since SSL use 20ms as frame shift...

Base_step=0.01 #base in sec
Frame_shifts= np.array([pow(2, i) for i in np.arange(Scale_num)])[SSL_shift:] 
Frame_shifts_list= [pow(2, i) for i in np.arange(Scale_num)][SSL_shift:] 

SCORE_SCALE = Frame_shifts[0]
LABEL_SCALE = 1
Multi_scales=Frame_shifts * Base_step #[0.01, 0.02, 0.04, 0.08, 0.16]

#EXTEND_SCALEs = [128, 256, 512, 1024]
EXTEND_SCALEs = []
UTTEER = False
ONLY_DIG = False


LAB2NUM={'genune':1.0, 'spoof':0.0} #float
SCORE_TYPE='mean'
SCORE_TYPE='min'
#default min.
#min, max.


#1. data load
##set work path
DSETs=['dev', 'eval']
#DSETs=['dev']
EXP_NAME="exp-2d-sametol"
RES_DIR="./results_"+SCORE_TYPE


CON_WOK_DIR = "/home/smg/zhanglin/workspace/PROJ/01spf-con/"
ASVDIR_NAME="01asvspoof-2d-multiscale"
special='202108'
MODEL_TYPEs = ['lstm-ap-noshare', 'pro_v2_ori']
#MODEL_TYPEs = ['pro_v1']
special='20210'
#MODEL_TYPEs = ['v2para-avgsepro-input_un', 'v2para-avgsepro-blstm_un', 'v2para-avgsepro-input_dwa', 
#        'v3headpara-beforeB_un', 'v3headpara-afterB_un']
##MODEL_TYPEs = ['v4unet-filter1_un', # no seed 02 
##MODEL_TYPEs = ['v4unet-filter2_un', 'v4unet-filter2_un-pcgrad', 'v4unet-filter2-noinputmse_un',
##        'v4unet-filter2-noinputmse_dwa', 'v4unet-filter2-noinputmse-sc124_un', 'v4unet3plus-filter2_un']
MODEL_TYPEs = ['v4unet-filter2-noinputmse-sc4816_un', 'v4unet-filter2-noinputmse-sc816_un']
MODEL_TYPEs = ['v4unet3plus-filter2_un']

MODEL_TYPEs = [ 'ori_un', 'ori_un-pcgrad', 'v1-avgsepro-input_un', 'v1-avgpro-blstm_un', 'v1-maxpro', 'v1-maxsepro']


MODEL_TYPEs = ['v4unet-filter2', 'v4unet-filter2-noinputmse-noutt_un']


#remove 
MODEL_TYPEs = ['v4unet-filter2-noinputmse_un', 
        'v4unet-filter2-noinputmse-sc124_un', 'v4unet-filter2-noinputmse-sc4816_un','v4unet-filter2-noinputmse-sc816_un',
        'v4unet-filter2-noinputmse-noutt_un']


#check ratio
CON_WOK_DIR = "/home/smg/zhanglin/workspace/PROJ/01spf-con/"
#/home/smg/zhanglin/workspace/PROJ/01spf-con/01asvspoof-2d-multiscale/00check/ratio/ratio-bal/v2para-avgsepro-input_un

asvdir_name="v4unet-filter2"
ASVDIR_NAME="01asvspoof-2d-multiscale/00check/ratio/"+asvdir_name
MODEL_TYPEs = ["ratio"+str(i) for i in np.arange(1,11)]
RES_DIR="./results_{}_checkratio/{}".format(SCORE_TYPE, asvdir_name)
special='2021'
#
#ASVDIR_NAME="01asvspoof-2d-multiscale/00check/balgenspf/"
#MODEL_TYPEs = ['v4unet-filter2', 'v2para-avgsepro-input_un']
#RES_DIR="./results_{}_checkbal".format(SCORE_TYPE)
#special='2021'
##
##
##ssl
#ASVDIR_NAME="01asvspoof-2d-multiscale-ssl/pret/"
#MODEL_TYPEs = ['ori-wav2vec2', 'ori-hubert']
#RES_DIR="results_{}_ssl-pre".format(SCORE_TYPE)
#special='2021'


#ASVDIR_NAME="01asvspoof-2d-multiscale-ssl/finet"
#MODEL_TYPEs = ['ori-hubert', 'ori-wav2vec2', 'ori-wav2vec2-last', 'ori-wav2vec2-last-lr', 'ori-wav2vec2-lr']
#MODEL_TYPEs = ['ori-wav2vec2', 'ori-wav2vec2-last',
#        'ori-wav2vec2-lr0005', 'ori-wav2vec2-lr0001', 
#        '1bl-wav2vec2', 'ori-max1l-wav2vec', '1bl-max1l-wav2vec2']
#MODEL_TYPEs = ['ori-wav2vec2-lr0001', 'ori-wav2vec2-lr0003', 'ori-wav2vec2-lr0005', 
#               'ori-wav2vec2-lr00001', 'ori-wav2vec2-lr000001']
##MODEL_TYPEs = ['1bl-max1l-wav2vec2', 'ori-wav2vec2-lr00005'] 
#MODEL_TYPEs = ['ori-wav2vec2-lr000005']
#RES_DIR="./results_{}_ssl_mulscale".format(SCORE_TYPE)
#special='2021'
#use_ext_flag = True #False # we only use a single seed now
##

#check finetune for each scale
#/home/smg/zhanglin/workspace/PROJ/01spf-con/01asvspoof-2d-multiscale/00check/ratio/ratio-bal/v2para-avgsepro-input_un
#/home/smg/zhanglin/workspace/PROJ/01spf-con/01asvspoof-2d-multiscale-ssl/finet-singlescale

#asvdir_name="ori-wav2vec2"
##asvdir_name="ori-wav2vec2-last"
#ASVDIR_NAME="01asvspoof-2d-multiscale-ssl/finet-singlescale/"+asvdir_name
#SCALEs=['1', '2', '4', '8', '16', 'utt']
##SCALEs=['1', '4', '8', '16', 'utt']
#MODEL_TYPEs = ["scale"+str(i) for i in SCALEs]
#RES_DIR="./results_{}_ssl-singlescale/{}".format(SCORE_TYPE, asvdir_name)
##special='2021'
#
#
###########scale utt
#/home/smg/zhanglin/workspace/PROJ/01spf-con/01asvspoof-2d-multiscale-ssl/finet-singlescale/scaleutt-w2v2

Ablation=['01lr', '02downsample', '03pool', '04postnet', '05loss', '06ssl']
idx=5

ASVDIR_NAME="01asvspoof-2d-multiscale-ssl/finet-singlescale/scaleutt/"+Ablation[idx]
ASVDIR_NAME="01asvspoof-2d-multiscale-ssl/FT-singlescale/scaleutt/"+Ablation[idx]
ALL_MODEL={}
ALL_MODEL['0']=[''] #lr
ALL_MODEL['1']=['LR00001-Dmax1lin-AP-Plin-Lp2s-Mw2v2', 'LR00001-Dmax-AP-Plin-Lp2s-Mw2v2']#02downsample
ALL_MODEL['2']=['LR00001-Dmax-AP-Plin', 'LR00001-Dmax-SAP-Plin']  #03pool
ALL_MODEL['5'] = ['LR00001-Dmax1lin-AP-P5gmlp-Lp2s-Mwavlmlarge']
MODEL_TYPEs=ALL_MODEL[f'{idx}']
RES_DIR="./results_{}_ssl-scaleutt/".format(SCORE_TYPE)



##multi scale 
#ssl
Ablation=['01lr', '02downsample', '03pool', '04postnet', '05loss', '06ssl']
idx = 5  ##index
ASVDIR_NAME="01asvspoof-2d-multiscale-ssl/FT-multiscale/" + Ablation[idx]
ALL_MODEL={}
ALL_MODEL['1'] = ['LR00001-Dmax1lin-AP-Plin-Lp2s-Mw2v2']


ALL_MODEL['5'] = ['LR00001-Dmax1lin-AP-P5gmlp-Lp2s-Mw2v2large', 'LR00001-Dmax1lin-AP-P5gmlp-Lp2s-Mw2v2',
        'LR00001-Dmax1lin-AP-P5gmlp-Lp2s-Mwavlmlarge']
ALL_MODEL['5'] = ['LR00001-Dmax1lin-AP-P5gmlp-Lp2s-Mwavlmlarge']
ALL_MODEL['5'] = ['LR00001-Dmax1lin-AP-P5gmlp-Lp2s-Mw2v2large']
ALL_MODEL['4'] = ['LR00001-Dmax1lin-AP-P5gmlp-LFocal-Mw2v2large']
#\
MODEL_TYPEs=ALL_MODEL[f'{idx}']
RES_DIR="./results_{}_ssl-FT-multiscale/".format(SCORE_TYPE)
#ASVDIR_NAME="01asvspoof-2d-multiscale-ssl-fusebalconspfdevnew/FT-multiscale/" + Ablation[idx]
#RES_DIR="./results_{}_fusebaldev/FT-multiscale/".format(SCORE_TYPE)
#special='20'


##single scale
#single_SCALE='utt'
#single_SCALE='32'
single_SCALE=sys.argv[1]
ASVDIR_NAME="01asvspoof-2d-multiscale-ssl/FT-singlescale/" + single_SCALE
#MODEL_TYPEs = ['LR00001-Dmax1lin-AP-P5gmlp-LFocal-Mw2v2large']
RES_DIR="./results_{}_ssl-FT-singlescale/{}/".format(SCORE_TYPE, single_SCALE)
ASVDIR_NAME="01asvspoof-2d-multiscale-ssl-fusebalconspfdev/FT-singlescale/" + single_SCALE
MODEL_TYPEs = ['LR00001-Dmax1lin-AP-P5gmlp-Lp2s-Mw2v2large']
#
#RES_DIR="./results_{}_fusebaldev/FT-singlescale/{}/".format(SCORE_TYPE,single_SCALE)
#RES_DIR="./results_{}_ssl-FT-singlescalemixup_sim/{}/".format(SCORE_TYPE, single_SCALE)

#ASVDIR_NAME="01asvspoof-2d-multiscale-ssl-mixup/FT-singlescale-mixup/" + single_SCALE
#RES_DIR="./results-lossmixup/FT-singlescale/{}/".format(single_SCALE)
#ASVDIR_NAME="01asvspoof-2d-multiscale-ssl-mixup/FT-singlescale-orimixup/" + single_SCALE
#RES_DIR="./results-labelmixup/FT-singlescale/{}/".format(single_SCALE)
special='20'
#
#
###multi scale bottomup-all
##FT_type='FT-topdown-fix'
#FT_type='FT-bottomup-all'
##MODEL = 'LR00001-Dmax1lin-AP-P5gmlp-Lp2s-MLFCC128-7'
#MODEL = 'LR00001-Dmax1lin-AP-P5gmlp-Lp2s-Mw2v2large'
#ASVDIR_NAME="01asvspoof-2d-multiscale-ssl/{}/{}".format(FT_type, MODEL) 
#MODEL_TYPEs = ['2', '4', '8', '16', '32', '64', 'utt']
##MODEL_TYPEs = ['64', 'utt']
###MODEL_TYPEs = ['utt']
#RES_DIR="./results_{}_ssl-{}_sim/{}/".format(SCORE_TYPE, FT_type,MODEL)
#special='2021'





#use_ext_flag = True
use_ext_flag = True
print_each = False

if( not os.path.exists(RES_DIR) ):
    os.makedirs(RES_DIR)

def load_scolab(file_path):
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
        gen=sco_lab_pd[sco_lab_pd.True_Class==LAB2NUM['genune']].Predict_Score_1 #bonafide score is Score_1
        spf=sco_lab_pd[sco_lab_pd.True_Class==LAB2NUM['spoof']].Predict_Score_1
    elif ('True_SpoofRatio' in sco_lab_pd.columns):
        sco_lab_pd['True_SpoofRatio']=sco_lab_pd['True_SpoofRatio'].astype(float)
        gen=sco_lab_pd[sco_lab_pd.True_SpoofRatio==0.0].Predict_Score_1
        spf=sco_lab_pd[sco_lab_pd.True_SpoofRatio>0.0].Predict_Score_1

    eer, threshold = eval_asvspoof.compute_eer(gen, spf)
    #print("EER: {:.3f}%".format(eer * 100))
    if(utteer_flag):
       
        return eer
        #tmp_pd = pd.DataFrame(columns=('Wavid', 'EER', 'Threshold'))
        #tmp_pd = tmp_pd.astype({"Wavid": str, "EER": float, "Threshold":float})
        #            }),ignore_index=True)
        #return tmp_pd

    return eer, threshold

def get_pd_in_scoretype(sco_lab_pd, fs, SCORE_TYPE='min', return_eer=False ):
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
        new_pd = new_pd.append(t, ignore_index=True)
#    if(return_eer):
#        eer, threshold = cal_eer_by_scolab(new_pd)    
#        return eer 

    return new_pd    

def get_eermatrix_scale(sco_lab_pd, scale, extend_scales = [], return_utteer=False):

    eers=np.zeros(len(Frame_shifts)) #* np.inf
    thresholds=np.zeros(len(Frame_shifts)) #* np.inf

    index = Frame_shifts.tolist().index(scale)
    for idx in np.arange(index, len(Frame_shifts)):
        if(idx==index):   
            # I error use idx == 0 to cal results.
            new_pd = sco_lab_pd
            if(return_utteer):
                utteer_pd = sco_lab_pd.groupby('Wavid').apply(cal_eer_by_scolab, utteer_flag=True)
                utteer_pd = pd.DataFrame(utteer_pd).reset_index()
                utteer_pd.rename( columns={0:'EER'}, inplace=True  )
        else:
            if(ONLY_DIG):
                continue
            fs = int(Frame_shifts[idx] / scale) 
            #we need to divide scale, since the further eer calculation is based on frame shift on basic scale.
            # any function passed to apply is applied along each column/series independent
            ######here we select min for score_0, and max for score_1(bonafide)

            # we only use Predict_score_1, so we only need to consider one column:
            #  and since here we use cosine similirity, so higher means more similar, Score_1 for bonafide. 
            #  -> smaller score means spoof.
            new_pd = sco_lab_pd.groupby('Wavid').apply(get_pd_in_scoretype, fs = fs, SCORE_TYPE = SCORE_TYPE)  #error
            #new_pd = get_pd_in_scoretype(sco_lab_pd, fs, SCORE_TYPE = SCORE_TYPE)
            #new_pd = sco_lab_pd.groupby('Wavid').apply(lambda x,fs,SCORE_TYPE: get_pd_in_scoretype(x, fs, SCORE_TYPE,))

        eers[idx], thresholds[idx] = cal_eer_by_scolab(new_pd)    

    if(len(extend_scales)>0):
        eers_ex=np.zeros(len(extend_scales)) #* np.inf
        thresholds_ex=np.zeros(len(extend_scales)) #* np.inf
        for idx, fs in enumerate(extend_scales):
            fs = int(fs/scale)
            new_pd = sco_lab_pd.groupby('Wavid').apply(get_pd_in_scoretype, fs = fs, SCORE_TYPE = SCORE_TYPE)  #error
            eers_ex[idx], thresholds_ex[idx] = cal_eer_by_scolab(new_pd)    

        if(return_utteer):
            return eers, thresholds, eers_ex, thresholds_ex, utteer_pd    
        return eers, thresholds, eers_ex, thresholds_ex, ''     

    if(return_utteer):
        return eers, thresholds, utteer_pd    

    return eers, thresholds, ''    

def get_utteer_by_seg(sco_lab_pd):
    eer_utt, threshold_utt = cal_eer_by_scolab(sco_lab_pd.groupby(['Wavid']).min())

    return eer_utt, threshold_utt


#def get_mean_multi_seed(eers_scales_seed, threshold_scales_seed):
#    eers_scales_seed_mean = np.zeros((len(Frame_shifts), len(Frame_shifts)+1)) #+1 is for utterance eer
#    threshold_scales_seed_mean =np.zeros((len(Frame_shifts), len(Frame_shifts)+1))
#    for seed_idx, (eer_s, threshold_s) in enumerate(zip(eers_scales_seed, threshold_scales_seed)):
#        eers_scales_seed_mean += eer_s.reshape(len(Frame_shifts), len(Frame_shifts)+1)
#        threshold_scales_seed_mean += threshold_s.reshape(len(Frame_shifts), len(Frame_shifts)+1)
#    seed_num = len(eers_scales_seed)    
#    eers_scales_seed_mean /= float(seed_num) 
#    threshold_scales_seed_mean /= float(seed_num) 
#
#    return eers_scales_seed_mean, threshold_scales_seed_mean

def get_mean_multi_seed(eer_scales_seed, threshold_scales_seed, return_var=False):
    eers_mean = np.zeros(eer_scales_seed[0].shape)
    threshold_mean = eers_mean.copy()
    for seed_idx, (eer_s, threshold_s) in enumerate(zip(eer_scales_seed, threshold_scales_seed)):
        eers_mean += eer_s
        threshold_mean += threshold_s

    seed_num = len(eer_scales_seed)    
    eers_mean = eers_mean/ float(seed_num) 
    threshold_mean = threshold_mean/ float(seed_num) 

    if(return_var):
        return eers_mean, threshold_mean

    return eers_mean, threshold_mean

#3. eer


def main():
    np.set_printoptions(linewidth=100000)
    for dset in DSETs:
        for idx, (mtype) in enumerate(MODEL_TYPEs):
            eers_scales_seed = []
            threshold_scales_seed = []
            eers_scales_seed_ex = []
            threshold_scales_seed_ex = []
            utteer_scales_seed = []
    
            modeldir=os.path.join(CON_WOK_DIR, ASVDIR_NAME, mtype)
    
            eer_file='{}/eer_{}_{}'.format(RES_DIR, mtype, dset)
    #        if(len(ASVDIR_NAME.split("/")) > 1):
    #            tmp_dir = '{}/{}'.format(RES_DIR, ASVDIR_NAME.split("/")[-1])
    #            if(os.path.exists(tmp_dir)):
    #                os.makedirs(tmp_dir) 
    #            eer_file='{}/eer_{}_{}'.format(tmp_dir, mtype, dset)
    
    
            if(os.path.exists(eer_file+'.npz') and (use_ext_flag)):
                res_data = np.load(eer_file+".npz", allow_pickle=True)
    
                eers_scales_seed_mean, threshold_scales_seed_mean = get_mean_multi_seed(
                        res_data['eers_scales_seed'], res_data['threshold_scales_seed'])
                if(len(res_data['eers_scales_seed_ex'])>0):
                    eers_ex_mean, threshold_ex_mean = get_mean_multi_seed(
                        res_data['eers_scales_seed_ex'], res_data['threshold_scales_seed_ex'])
    
                print("="*10,"{}_{} in the score type {}, seed {}".format(mtype,dset,SCORE_TYPE, len(eers_scales_seed)),"="*10)
                if(print_each):
                    print('-'*20+'eer for each res')
                    print(res_data['eers_scales_seed']*100)
                    print('-'*20+'threshold for each res')
                    print(res_data['threshold_scales_seed'])
                if(len(EXTEND_SCALEs)>0):
                    print("-"*10,"{}".format(EXTEND_SCALEs),"="*10)
                    print("eer:\n", eers_ex_mean * 100)

                print('-'*20+'mean')
    
    
                print(eers_scales_seed_mean * 100)
                if(len(EXTEND_SCALEs)>0):
                    print(eers_ex_mean * 100)
                #print(threshold_scales_seed_mean)
            else:
                for sml_dir in os.listdir(modeldir): #read all small model in this dir(seed)
                    eer_scales=[]
                    threshold_scales=[]
                    eers_scales_ex=[]
                    thresholds_scales_ex=[]
                    utteer_scales = {}
        
                    model_out_dir=os.path.join(modeldir, sml_dir,'output_con_ali')
                    if(sml_dir.find('base')<0 and sml_dir.find('tmp')<0 and sml_dir.find('old')<0 and  os.path.exists(model_out_dir)):
                        #to judge wether we have aligned score .pkl file.
                        #and this is not temperate folder.
                        find_it=False
                        while(True):
                            #to find the time_tag based on the ali.npz.
                            for score_name in os.listdir(model_out_dir):
                                name, app = os.path.splitext(score_name)
                                if ( name.find(special)>=0 and name.find('score_ali')>=0 and name.find(dset)>=0  and app == '.pkl'):
                                    find_it=True
                                    break
                            if(find_it):
                                break
                            print("no score_ali file, please check " + model_out_dir)
                            sys.exit()
            
                        time_tag=name.split('_')[-1]
                        for scale in Frame_shifts:
                            sco_ali_file=os.path.join(model_out_dir,
                                    '{}_score_ali_{}_{}.pkl'.format(dset,scale,time_tag))
                            sco_lab_pd = load_scolab(sco_ali_file)
    
                            #segment eer
                            if(len(EXTEND_SCALEs)>0):
                                eers, thresholds, eers_ex, thresholds_ex, utteer_pd = get_eermatrix_scale(
                                        sco_lab_pd, scale, extend_scales = EXTEND_SCALEs, return_utteer=UTTEER)
                                eers_scales_ex.append(eers_ex)
                                thresholds_scales_ex.append(thresholds_ex)
                            else:
                                eers, thresholds, utteer_pd = get_eermatrix_scale(
                                        sco_lab_pd, scale, extend_scales = EXTEND_SCALEs, return_utteer=UTTEER)
    
                            #utterance eer from segments score
                            eer_utt, threshold_utt = get_utteer_by_seg(sco_lab_pd)
    
                            #append eer(segment eer in x Scale_num; utterance eer)
                            eer_scales.append(np.concatenate((eers, [eer_utt])))
                            threshold_scales.append(np.concatenate((thresholds, [threshold_utt])))
                            utteer_scales[f'{scale}']=utteer_pd
    
                        eers_scales_seed.append(np.concatenate(eer_scales).reshape(len(Frame_shifts), -1))    
                        threshold_scales_seed.append(np.concatenate(threshold_scales).reshape(len(Frame_shifts), -1))
                        utteer_scales_seed.append(utteer_scales)
    
                        if(len(EXTEND_SCALEs)>0):
                            eers_scales_seed_ex.append(np.concatenate(eers_scales_ex).reshape(len(Frame_shifts), -1))
                            threshold_scales_seed_ex.append(np.concatenate(thresholds_scales_ex).reshape(len(Frame_shifts), -1))
        
                np.savez(eer_file, 
                        eers_scales_seed = eers_scales_seed, threshold_scales_seed = threshold_scales_seed,
                        eers_scales_seed_ex = eers_scales_seed_ex, threshold_scales_seed_ex = threshold_scales_seed_ex,
                        utteer_scales_seed = utteer_scales_seed)
                #final column is utterance eer from segments score.
        
                eers_scales_seed_mean, threshold_scales_seed_mean = get_mean_multi_seed(
                        eers_scales_seed, threshold_scales_seed)
                print("="*10,"{}_{}".format(mtype,dset),"="*10)
                print("eer:\n", eers_scales_seed_mean * 100)
                #print("threshold:\n", threshold_scales_seed_mean)
                if(len(EXTEND_SCALEs)>0):
                    eers_ex_mean, threshold_ex_mean = get_mean_multi_seed(
                        eers_scales_seed_ex, threshold_scales_seed_ex)
                    print("="*10,"{}".format(EXTEND_SCALEs),"="*10)
                    print("eer:\n", eers_ex_mean * 100)
    
if __name__ == "__main__": 
    main()

