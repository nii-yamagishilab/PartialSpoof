#!/usr/bin/env python

# Copyright 2021 National Institute of Informatics (author: Xin Wang, wangxin@nii.ac.jp)
# Copyright 2023 National Institute of Informatics (author: Lin Zhang, zhanglin@nii.ac.jp)
# Licensed under the BSD 3-Clause License.

"""
model.py

Self defined model definition.
Usage:

"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torchaudio
import torch.nn.functional as torch_nn_func

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end
import core_scripts.other_tools.debug as nii_debug
import core_scripts.data_io.seq_info as nii_seq_tk
import core_modules.p2sgrad as nii_p2sgrad

from multi_scale.post import MaxPool1dLin_gmlp_scales
#s3prl
import s3prl.hub as hub
device = 'cuda'

##############
## util
##############
PS_PATH="/home/smg/zhanglin/workspace/PROJ/Public/CODE/PartialSpoof" #path to PartialSpoof

Scale_num=7   
SSL_shift=1   ##since SSL use 20ms as frame shift, we start from 1. and 0 is for 10 ms...

Base_step=0.01 #in sec
Frame_shifts= np.array([pow(2, i) for i in np.arange(Scale_num)])[SSL_shift:] 
Frame_shifts_list= [pow(2, i) for i in np.arange(Scale_num)][SSL_shift:] 

LABEL_SCALE = 1
Multi_scales=Frame_shifts * Base_step #[0.01, 0.02, 0.04, 0.08, 0.16]

ASVSPOOF_PROTOCAL=PS_PATH+'/project-NN-Pytorch-scripts.202102/project/02-asvspoof/DATA/asvspoof2019_LA/protocol.txt' #protocal of asvspoof2019


hidd_dims ={'wav2vec':512, 'wav2vec2':768, 'hubert':768, 'wav2vec2_xlsr':1024, 'wavlm_base_plus':768, 'wav2vec2_local':1024}
ssl_model='wav2vec2_local'
ssl_ckpt=PS_PATH+'/modules/ssl_pretrain/w2v_large_lv_fsh_swbd_cv.pt'


def protocol_parse(protocol_filepath):
    """ Parse protocol of ASVspoof2019 and get bonafide/spoof for each trial
    
    input:
    -----
      protocol_filepath: string, path to the protocol file
        for convenience, I put train/dev/eval trials into a single protocol file
    
    output:
    -------
      data_buffer: dic, data_bufer[filename] -> 1 (bonafide), 0 (spoof)
    """ 
    data_buffer = {}
    temp_buffer = np.loadtxt(protocol_filepath, dtype='str')
    for row in temp_buffer:
        if row[-1] == 'bonafide':
            data_buffer[row[1]] = 1
        else:
            data_buffer[row[1]] = 0
    return data_buffer

def protocol_parse_con(reco2seglabel_filepath):
    """ Get label fro PartialSpoof database
    
    input:
    -----
      reco2seglabel_filepath: npy, path to the label file
    
    output:
    -------
      data_buffer: dict{list}, data_bufer[filename] -> 1 (bonafide), 0 (spoof)
    """ 
    data = np.load(reco2seglabel_filepath, allow_pickle=True)
    data_buffer=data.item()
    return data_buffer
##############
## FOR MODEL
##############

class Model(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, mean_std=None):
        super(Model, self).__init__()

        ##### required part, no need to change #####

        #output_dir
        self.output_dir = args.output_dir #default
        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         args, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        
        # Load protocol and prepare the target data for network training
        if(args.data_type == 'asvspoof'):
            protocol_file = ASVSPOOF_PROTOCAL
            self.protocol_parser = protocol_parse(protocol_file)
        else:
            reco2spfseglab_file = args.temp_flag
            self.protocol_parser_con = protocol_parse_con(reco2spfseglab_file)
            if(args.temp_flag_dev):
                reco2spfseglab_file_dev = args.temp_flag_dev
                self.protocol_parser_condev = protocol_parse_con(reco2spfseglab_file_dev)
        
        self.len_limit_unit = 16 #base on the configuration of network.

        # Working sampling rate
        #  torchaudio may be used to change sampling rate
        self.m_target_sr = 16000

        # frame shift (number of waveform points)
        self.frame_hops = [160]
        # frame length
        self.frame_lens = [320]
        # FFT length
        self.fft_n = [512]

        # LFCC dim (base component)
        self.lfcc_dim = [20]
        self.lfcc_with_delta = True

        # window type
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating (not used)
        self.amp_floor = 0.00001
        
        # number of frames to be kept for each trial
        # no truncation
        self.v_truncate_lens = [None for x in self.frame_hops]


        # number of sub-models (by default, a single model)
        self.v_submodels = len(self.frame_lens)        

        # dimension of embedding vectors
        self.v_emd_dim = -2

        # output classes
        self.v_out_class = 2

        #squeeze-ro-extication
        self.reduction = 2 
        self.m_se_pooling = []
        ####
        # create network
        ####
        # 1st part of the classifier
        self.m_transform = []
        # 
        ## 2nd part of the classifier
        self.m_angle_scales = torch_nn.ModuleDict()

        #weight for hidden_states:
        self.weight_hidd = torch_nn.Parameter(torch.zeros(30, device='cuda'))
        self.hidden_features_dim = hidd_dims[ssl_model] #768

        self.extracter = getattr(hub, ssl_model)(ssl_ckpt)

        self.ssl_finetune = args.ssl_finetune
        self.multi_scale_active = args.multi_scale_active
        self.multi_branch_fix = args.multi_branch_fix

        self.data_type = args.data_type

        for idx, (trunc_len, fft_n, lfcc_dim) in enumerate(zip(
                self.v_truncate_lens, self.fft_n, self.lfcc_dim)):
            
            fft_n_bins = fft_n // 2 + 1
            if self.lfcc_with_delta:
                lfcc_dim = lfcc_dim * 3
            
            self.m_transform.append(
                    MaxPool1dLin_gmlp_scales(num_scale = len(Frame_shifts), feature_F_dim = self.hidden_features_dim, 
                        emb_dim=self.v_emd_dim, seq_len=2001,
                        gmlp_layers=5, batch_first=True, flag_pool='ap')
            )

            if(self.v_emd_dim > 0):
                for fs_i in Frame_shifts:
                    self.m_angle_scales[f"{fs_i}"]= nii_p2sgrad.P2SActivationLayer(self.v_emd_dim, self.v_out_class)
            elif(self.v_emd_dim < 0):
                for i, fs_i in enumerate(Frame_shifts):  # i here indicate how many downsample will happen
                    self.m_angle_scales[f"{fs_i}"]= nii_p2sgrad.P2SActivationLayer(
                            int(self.hidden_features_dim/pow(abs(self.v_emd_dim),i+1)), self.v_out_class)

        self.m_debugger_segscore_ali = []
        for i in Frame_shifts:
            self.m_debugger_segscore_ali.append(nii_debug.data_probe())

        self.m_transform = torch_nn.ModuleList(self.m_transform)
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, data_mean_std=None):
        """ prepare mean and std for data processing
        This is required for the Pytorch project, but not relevant to this code
        """
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            if in_m.shape[0] != in_dim or in_s.shape[0] != in_dim:
                print("Input dim: {:d}".format(in_dim))
                print("Mean dim: {:d}".format(in_m.shape[0]))
                print("Std dim: {:d}".format(in_s.shape[0]))
                print("Input dimension incompatible")
                sys.exit(1)
            if out_m.shape[0] != out_dim or out_s.shape[0] != out_dim:
                print("Output dim: {:d}".format(out_dim))
                print("Mean dim: {:d}".format(out_m.shape[0]))
                print("Std dim: {:d}".format(out_s.shape[0]))
                print("Output dimension incompatible")
                sys.exit(1)
        else:
            in_m = torch.zeros([in_dim])
            in_s = torch.ones([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.ones([out_dim])
            
        return in_m, in_s, out_m, out_s
        
    def normalize_input(self, x):
        """ normalizing the input data
        This is required for the Pytorch project, but not relevant to this code
        """
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """ normalizing the target data
        This is required for the Pytorch project, but not relevant to this code
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        This is required for the Pytorch project, but not relevant to this code
        """
        return y * self.output_std + self.output_mean

    ##/home/smg/zhanglin/workspace/06ssl/s3prl/s3prl/upstream/interfaces.py : 212
    def _weighted_sum(self, feature):
        layer_num = len(feature)
       # assert self.layer_num == len(feature), (
       #     "If you run into this error, there is a great chance"
       #     " you are finetuning the upstream with wav2vec2's transformer blocks"
       #     " in weighted-sum mode (default), including wav2vec2, hubert, and decoar2."
       #     " These models use the layerdrop technique which causes the different number"
       #     " of layer forwards between different model forwards, resulting in different"
       #     " number of hidden states for different model forwards. Hence, finetuning"
       #     " these upstreams is essentially incompatible with weight-sum mode unless"
       #     " you turn off the layerdrop option in fairseq. See:"
       #     " https://github.com/pytorch/fairseq/blob/f6abcc2a67328bee8b15c596bb626ce2d720aae6/fairseq/models/wav2vec/wav2vec2.py#L857"
       #     " However, since finetuning upstreams will backward the gradient through all layers"
       #     " which serves the same functionality as weighted-sum: all layers can be used for different"
       #     " downstream tasks. Hence instead of finetuning upstream with weighted-sum, we suggest to"
       #     " follow the more common setting: finetuning upstream with the last layer. Please use the"
       #     " following options: --upstream_trainable --upstream_feature_selection last_hidden_state."
       #     " Or: -f -s last_hidden_state"
       # )
        stacked_feature = torch.stack(feature, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(layer_num, -1)
        norm_weights = torch_nn.functional.softmax(self.weight_hidd[:layer_num], dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def _front_end(self, wav, idx, trunc_len, datalength):
        """ simple fixed front-end to extract features
        
        input:
        ------
          wav: waveform convert to list for s3prl
          idx: idx of the trial in mini-batch
          trunc_len: number of frames to be kept after truncation
          datalength: list of data length in mini-batch

        output:
        -------
          x_reps: front-end featues, (batch, frame_num, frame_feat_dim)
        """
        x = [i for i in wav.squeeze(2)]
         
        if(self.ssl_finetune):
            x_reps = self.extracter(x)["hidden_states"]
        else:
            with torch.no_grad():
                x_reps = self.extracter(x)["hidden_states"]

        feature = self._weighted_sum(x_reps)

        return feature

    def _compute_embedding(self, x, datalength, filenames, eval_flag=False):
        """ definition of forward method 
        Assume x (batchsize, length, dim)
        Output x (batchsize * number_filter, output_dim)
        """
        # resample if necessary
        #x = self.m_resampler(x.squeeze(-1)).unsqueeze(-1)
        
        # number of sub models
        batch_size = x.shape[0]

        # compute scores for each sub-models
        for idx, (fs, fl, fn, trunc_len, m_trans ) in \
            enumerate(
                zip(self.frame_hops, self.frame_lens, self.fft_n, 
                    self.v_truncate_lens, self.m_transform)): 
#                    self.m_before_pooling)):
            
            # extract front-end feature
            x_reps = self._front_end(x, idx, trunc_len, datalength)

            # compute scores
            #  1. unsqueeze to (batch, 1, frame_length, fft_bin)
            #  2. compute hidden features
            hidden_features_scales = m_trans(x_reps)



        return hidden_features_scales

    def _compute_score(self, x, inference=False):
        """
        utterance score.
        """
        # number of sub models
        batch_size = x.shape[0]

        # compute score through p2sgrad layer
        out_score = torch.zeros(
            [batch_size * self.v_submodels, self.v_out_class], 
            device=x.device, dtype=x.dtype)

        #compute score for each submodels
        for idx in range(self.v_submodels):
            tmp_score = self.m_angle_scales[f'{Frame_shifts[-1]}'](x[idx * batch_size : (idx+1) * batch_size])
            out_score[idx * batch_size : (idx+1) * batch_size] = tmp_score

        if inference:
            # output_score [:, 1] corresponds to the positive class
            return out_score[:, 1]
        else:
            return out_score

    def _compute_seg_score_scales(self, x_scales, filenames, inference=False):
        """
        """
        # compute score through p2sgrad layer
        out_score_scales = {}
        for idx in range(len(x_scales)): #initialize the output
            out_score_scales[f'{Frame_shifts[idx]}'] =[]

        # compute scores for each scales
        for idx, x in enumerate(x_scales):
            batch_size = x.shape[0]
            feat_dim = x.shape[-1]

            # score flatten to (batch_size * seq_length, feat_dim)
            score = self.m_angle_scales[f'{Frame_shifts[idx]}'](x.view(-1, feat_dim))
            out_score_scales[f'{Frame_shifts[idx]}'].append(score)

        if inference:
            return [out_score[:, 1] for key,out_score in out_score_bag.items()]
        else:
            return out_score_scales

    def _get_target(self, filenames):
        try:
            return [self.protocol_parser[x] for x in filenames]
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)

    def lab_in_scale(self, labvec, shift):
        num_frames = int(len(labvec) / shift)
        if(num_frames==0):
            #for some case that duration < frameshift. 20211031. the case in 0.64 scale. 
            num_frames=1
            new_lab = np.zeros(num_frames, dtype=int)
            new_lab[0] = min(labvec) #only for binary '0' for spoof
        else:
            #common case.
            new_lab = np.zeros(num_frames, dtype=int)
            for idx in np.arange(num_frames):
                st, et  = int(idx * shift), int((idx+1)*shift)
                new_lab[idx] = min(labvec[st:et]) #only for binary '0' for spoof
        return new_lab


    def _get_segcon_target(self, filename, shift, dev_flag=False):
        res=[]
        labvecs=[]
        try:
           if(dev_flag):
               labvec = self.protocol_parser_condev[filename]
           else:
               labvec = self.protocol_parser_con[filename]
           
           if(shift > LABEL_SCALE):
               labvec =self.lab_in_scale(labvec, shift) 

           return np.array(labvec,dtype=int)

        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)

    def _get_con_target(self, filenames, dev_flag=False):

        res=[]
        labvecs=[]
        try:
            for x in filenames:
                if(dev_flag):
                    labvec=self.protocol_parser_condev[x]
                else:
                    labvec=self.protocol_parser_con[x]

                labvecs.append(labvec)
               # res.append(str(1-int('0' in labvec)))   
                if '0' in labvec: #spoof
                    res.append(0)
                else:             #bonafide 
                    res.append(1)
            return np.array(res, dtype=int)    
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)



    def _prepare_segcon_target_ali(self, filenames, score_scales, dev_flag=False):
        #get_con_target may have different dim with score, so we use prepare.

        gpu_num = torch.cuda.device_count()
        batch_size=int(len(filenames)/ gpu_num)  #/2 here is for DataParallel
        class_num=score_scales[f'{Frame_shifts[0]}'].shape[1]
        new_score_scales = {}
        target_vec_scales = {}

        #time omputation, I don know how to iterate list in dict at the same time 
        for shift, scores in score_scales.items():
            #print(scores.shape)
            #seq_len = int(scores.shape[0] /batch_size) #DataParallel do not accept reshape(-1) .... 
            #actually since it is put on two gpus, the dim will be reduced by twice
            scores = scores.reshape(batch_size, -1,class_num)
            new_score_scales[shift] = []
            target_vec_scales[shift] = []

            # target
            target_vec_scales[shift] = []
            # for each sub model

            for idx, (x, score) in enumerate(zip(filenames, scores)):
                target = self._get_segcon_target(x, float(shift), dev_flag)
                # repeat this number of times for each trial
                time_len = score.shape[0] 
                # (batchsize, feat_len)
                target_vec = torch.tensor(
                    target[:time_len], device=score.device, dtype=score.dtype)
                # (batchsize * feat_len, 1)
                target_vec_scales[shift].append(target_vec.reshape([-1, 1]))
                #align:
                new_score_scales[shift].append(score[:target_vec.shape[0]])
                
            new_score_scales[shift] = torch.cat(new_score_scales[shift])    
            target_vec_scales[shift] = torch.cat(target_vec_scales[shift])

        return new_score_scales, target_vec_scales


    def forward(self, x, fileinfo, dev_flag=False):
        

        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        batch_size=len(filenames)
        if self.training:
            
            outs = self._compute_embedding(x, datalength, filenames) 
            #[seg1, seg2, ..., utt] [B x T x emb_d, B x T/2xd,...,B x emb_d ]
            feature_vec, seg_feature_scales = outs[-1],outs[:-1] 

            #utt
            utt_scores = self._compute_score(feature_vec)
            # target
            target = self._get_con_target(filenames, dev_flag)
            target_vec = torch.tensor(target, 
                                      device=x.device, dtype=utt_scores.dtype)
            utt_target_vec = target_vec.repeat(self.v_submodels)

            if( ['utt'] == self.multi_scale_active ): 
                # only one utt in the scales
                seg_score_scales = ''
                seg_target_scales = ''
                return [seg_score_scales, seg_target_scales, utt_scores, utt_target_vec, str(True) ] # TypeError: 'bool' object is not iterable # DataParallel


            ###pegrepare segmental cos score and target 
            seg_score_scales = self._compute_seg_score_scales(seg_feature_scales, filenames)
            for scale, score in seg_score_scales.items():
                seg_score_scales[f'{scale}'] =  torch.cat(score, dim=0)

            #segment
            seg_score_scales, seg_target_scales = self._prepare_segcon_target_ali(filenames, seg_score_scales, dev_flag)


            
            return [seg_score_scales, seg_target_scales, utt_scores, utt_target_vec, str(True) ] # TypeError: 'bool' object is not iterable # DataParallel

        else:
            if ( any( len < self.frame_hops[0] * self.len_limit_unit for len in datalength) ):
                # 16 is based on the number of maxpooling
                #only cover one-dim in batch. [1, *, *]
                return None
            outs = self._compute_embedding(x, datalength, filenames, True) 
            #[seg1, seg2, ..., utt] [B x T x emb_d, B x T/2xd,...,B x emb_d ]
            feature_vec, seg_feature_scales = outs[-1],outs[:-1] 

            ###pegrepare segmental cos score and target 
            seg_score_scales = self._compute_seg_score_scales(seg_feature_scales, filenames)
            for scale, score in seg_score_scales.items():
                seg_score_scales[f'{scale}'] =  torch.cat(score, dim=0)
            #utt
            utt_scores = self._compute_score(feature_vec)

            if(self.data_type=='asvspoof'):
                target = self._get_target(filenames)

            else:
                seg_score_scales, seg_target_scales = self._prepare_segcon_target_ali(filenames, seg_score_scales, dev_flag)

                #save target....
                for idx, (scale, score_ali) in enumerate(seg_score_scales.items()):
                    score_ali = score_ali.cpu().detach().numpy()
                    target_ali = seg_target_scales[scale].cpu().detach().numpy()
                    filename_ali = np.array(filenames * score_ali.shape[0]).reshape(-1,1)

                    ali = np.concatenate((filename_ali, score_ali, target_ali),axis=1)
                    np.concatenate((filename_ali, target_ali, score_ali),axis=1)

                    self.m_debugger_segscore_ali[idx].add_data(ali)

                target = self._get_con_target(filenames, dev_flag)

            #print utt score in different scale
            print("Output, %s, %d" % (filenames[0], target[0]) ,end="") 
            for s_idx in Frame_shifts:
                print(", %f" % seg_score_scales[f'{s_idx}'][:,1].min(), end="")
            print(", %f" % utt_scores[:,1])

            # don't write output score as a single file
            return None

    def finish_up_inference(self):
        for i, fs_i in enumerate(Frame_shifts):
            if(self.data_type=='partialspoof'):
                self.m_debugger_segscore_ali[i].dump('{}_score_ali_{}'.format(self.output_dir, str(fs_i)))
        return

class Loss():
    """ Wrapper to define loss function 
    """
    def __init__(self, args):
        """
        """
        self.m_loss = nii_p2sgrad.P2SGradLoss()
        self.multi_scale_active = args.multi_scale_active
        self.task_num = len(self.multi_scale_active)

    def compute(self, outputs, target):
        """ 
        """
        tol_loss = []

        #check segmental loss
        if(outputs[0]!='' and outputs[1]!=''):
            #check which scale will used to calculate losses
            for key, score in outputs[0].items():
                if(key in self.multi_scale_active):
                    seg_loss = self.m_loss(score, outputs[1][key])
                    if(self.task_num>1):  #multi-task
                        tol_loss.append(seg_loss)
                    if(self.task_num==1):  #single-task
                        single_loss = seg_loss
                else:
                    pass

        #check utt loss
        if( 'utt' in self.multi_scale_active):
            utt_loss = self.m_loss(outputs[2], outputs[3])
            if( self.task_num>1):
                tol_loss.append(utt_loss)
            elif(self.task_num==1):
                single_loss = utt_loss
            else:
                pass

        #return loss
        if(self.task_num >1 ):
            return [tol_loss, [True for x in range(len(tol_loss)) ]]
        elif(self.task_num == 1):
            return single_loss
        else:
            raise NotImplementedError

    
if __name__ == "__main__":
    print("Definition of model")


