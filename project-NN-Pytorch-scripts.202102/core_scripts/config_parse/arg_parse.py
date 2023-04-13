#!/usr/bin/env python
# Copyright 2021 National Institute of Informatics (author: Xin Wang, wangxin@nii.ac.jp)
# Copyright 2023 National Institute of Informatics (author: Lin Zhang, zhanglin@nii.ac.jp)
# Licensed under the BSD 3-Clause License.

"""
config_parse

Argument parse

"""
from __future__ import absolute_import

import os
import sys
import argparse

import core_scripts.other_tools.list_tools as nii_list_tools
import core_scripts.other_tools.display as nii_display


#############################################################
# argparser
#
def f_args_parsed(argument_input = None):
    """ Arg_parse
    """
    
    parser = argparse.ArgumentParser(
        description='General argument parse')
    
    ######
    # lib
    mes = 'module of model definition (default model, model.py will be loaded)'
    parser.add_argument('--module-model', type=str, default="model", help=mes)

    mes = 'module of configuration (default config, config.py will be loaded)'
    parser.add_argument('--module-config', type=str, default="config", 
                        help=mes)
    
    ######
    # Training settings    
    mes = 'batch size for training/inference (default: 1)'
    parser.add_argument('--batch-size', type=int, default=1, help=mes)
    
    mes = 'number of epochs to train (default: 50)'
    parser.add_argument('--epochs', type=int, default=50, help=mes)
    
    mes = 'number of no-best epochs for early stopping (default: 5)'
    parser.add_argument('--no-best-epochs', type=int, default=5, help=mes)

    mes = 'sampler (default: None). Default sampler is random shuffler'
    mes += 'default'
    parser.add_argument('--sampler', type=str, default='None', help=mes)

    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='learning rate (default: 0.0001)')
    
    mes = 'learning rate decaying factor, using '
    mes += 'torch.optim.lr_scheduler.ReduceLROnPlateau(patience=no-best-epochs,'
    mes += ' factor=lr-decay-factor). By default, no decaying is used.'
    mes += ' Training stopped after --no-best-epochs.'
    parser.add_argument('--lr-decay-factor', type=float, default=-1.0, help=mes)
    

    mes = 'L2 penalty on weight (default: not use). '
    mes += 'It corresponds to the weight_decay option in Adam'
    parser.add_argument('--l2-penalty', type=float, default=-1.0, help=mes)

    mes = ''
    mes += 'It corresponds to the eps option in Adam'
    parser.add_argument('--eps', type=float, default=1e-8, help=mes)

    mes = 'gradient norm (torch.nn.utils.clip_grad_norm_ of Pytorch)'
    mes += 'default (-1, not use)'
    parser.add_argument('--grad-clip-norm', type=float, default=-1.0,
                        help=mes)

    mes = 'lr scheduler: 0: ReduceLROnPlateau (default); 1: StepLR; '
    mes += 'this option is set on only when --lr-decay-factor > 0. '
    mes += 'Please check core_scripts/op_manager/lr_scheduler.py '
    mes += 'for detailed hyper config for each type of lr scheduler'
    parser.add_argument('--lr-scheduler-type', type=int, default=0, help=mes)

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed (default: 1)')
    
    mes = 'turn model.eval() on validation set (default: false)'
    parser.add_argument('--eval-mode-for-validation', \
                        action='store_true', default=False, help=mes)

    mes = 'if model.forward(input, target), please set this option on. '
    mes += 'This is used for autoregressive model, auto-encoder, and so on. '
    mes += 'When --model-forward-with-file-name is also on, '
    mes += 'model.forward(input, target, file_name) should be defined'
    parser.add_argument('--model-forward-with-target', \
                        action='store_true', default=False, help=mes)

    mes = 'if model.forward(input, file_name), please set option on. '
    mes += 'This is used with forward requires file name of the data. '
    mes += 'When --model-forward-with-target is also on, '
    mes += 'model.forward(input, target, file_name) should be defined'
    parser.add_argument('--model-forward-with-file-name', \
                        action='store_true', default=False, help=mes)
    
    mes = 'shuffle data? (default true). Set --shuffle will turn off shuffling'
    parser.add_argument('--shuffle', action='store_false', \
                        default=True, help=mes)

    mes = 'number of parallel workers to load data (default: 0)'
    parser.add_argument('--num-workers', type=int, default=0, help=mes)

    mes = 'use DataParallel to levarage multiple GPU (default: False)'
    parser.add_argument('--multi-gpu-data-parallel', \
                        action='store_true', default=False, help=mes)

    mes = 'way to concatenate multiple datasets: '
    mes += 'concatenate: simply merge two datasets as one large dataset. '
    mes += 'batch_merge: make a minibatch by drawing one sample from each set. '
    mes += '(default: concatenate)'
    parser.add_argument('--way-to-merge-datasets', type=str, \
                        default='concatenate', help=mes)
    ######
    # options to save model / checkpoint
    parser.add_argument('--save-model-dir', type=str, \
                        default="./", \
                        help='save model to this direcotry (default ./)')
    
    mes = 'do not save model after every epoch (default: False)'
    parser.add_argument('--not-save-each-epoch', action='store_true', \
                        default=False, help=mes)

    mes = 'name prefix of saved model (default: epoch)'
    parser.add_argument('--save-epoch-name', type=str, default="epoch", \
                        help=mes)

    mes = 'name of trained model (default: trained_network)'
    parser.add_argument('--save-trained-name', type=str, \
                        default="trained_network", help=mes)
    
    parser.add_argument('--save-model-ext', type=str, default=".pt",
                        help='extension name of model (default: .pt)')
    
    #######
    # options to load model
    mes = 'a trained model for inference or resume training '
    parser.add_argument('--trained-model', type=str, \
                        default="", help=mes + "(default: '')")

    mes = 'do not load previous training error information.'
    mes += " Load only model para. and optimizer state  (default: false)"
    parser.add_argument('--ignore-training-history-in-trained-model', 
                        action='store_true', \
                        default=False, help=mes)    

    mes = 'do not load previous training statistics in optimizer.'
    mes += " (default: false)"
    parser.add_argument('--ignore-optimizer-statistics-in-trained-model', 
                        action='store_true', \
                        default=False, help=mes)    

    mes = 'run inference mode (default: False, run training script)'
    parser.add_argument('--inference', action='store_true', \
                        default=False, help=mes)    
    #######
    # options to output
    mes = 'path to save generated data (default: ./output)'
    parser.add_argument('--output-dir', type=str, default="./output", \
                        help=mes)
    mes = 'which optimizer to use (Adam | SGD, default: Adam)'
    parser.add_argument('--optimizer', type=str, default='Adam', help=mes)
    
    mes = 'verbose level 0: nothing; 1: print error per utterance'
    mes = mes + ' (default: 1)'
    parser.add_argument('--verbose', type=int, default=1,
                        help=mes)

    #######
    # options for user defined 
    #zlin add the flag for dev
    
    #mes = 'used to transfer label to target level (for rttm file multi-class). '
    #parser.add_argument('--othlab', choices=['attckid', 'input', 'inputpro', \
    #                                         'duration', 'conversion', 'spkpre', 'outputs'], \
    #                    nargs='+', default=['attckid'], help=mes)


    mes = 'a temporary flag without specific purpose.'
    mes += 'User should define args.temp_flag only for temporary usage.'
    parser.add_argument('--temp-flag', type=str, default='', help=mes)

    mes = 'a temporary flag without specific purpose.'
    mes += 'User should define args.temp_flag only for temporary usage.'
    parser.add_argument('--temp-flag-dev', type=str, default='', help=mes)

    mes = 'a tmpeorary flag for dev, as dev should use another protocol. default: false)'
    parser.add_argument('--dev-flag', \
                        action='store_true', default=False, help=mes)

    #######
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)')    


    # zlin add loss weight stratigies
    mes = 'apply weight strategies for loss of multi-task learning)'
    parser.add_argument('--uncertainty-weight', action='store_true', \
                        default=False, help=mes)

    # or maybe we need to choose?
    parser.add_argument('--loss-weight', choices=['none', 'uncertainty', 'dwa'], \
            default='none', help=mes)

    parser.add_argument('--loss-weight-hyper', type=float, default=2.0, 
                        help='hyper-parameters for loss weight (default: 2.0)')
    #DWA: 2.0, tempature.

    mes = 'apply PCGrad'
    parser.add_argument('--PCGrad', action='store_true', \
                        default=False, help=mes)

    #ssl:
    mes = 'Do you want to fine-tune ssl'
    parser.add_argument('--ssl-finetune', action='store_true', \
                        default=False, help=mes)

    mse = 'how to select hidden feature from ssl model, maybe the last layer or weighted all layers?'
    parser.add_argument('--hidden-feature-selection', choices=['all', 'last'], \
            default='all', help=mes)


    #single-tasl for different scale / utt
    mse = 'This is only for single task. only implenment for ssl'
    parser.add_argument('--single-task', action='store_true', \
                        default=False, help=mes)

    parser.add_argument('--task-id', choices=['1', '2', '4', '8', '16', 'utt'], \
            default='utt', help=mes)

    
    #set multi task for select scale and gradually training 
    mse = "This is for user-defined multi-scale. (I write here for gradually train.)"
    parser.add_argument('--multi-scale-active', type=str, nargs='+', \
                        default=['2', '4','8','16', '32', '64', 'utt'], help=mes)

    mse = """This is for user-defined multi-scale, set some branch fixed during training.
           --multi-branch-fixed ['utt', '16'] ..
           (I write here for gradually train.)"""
    parser.add_argument('--multi-branch-fix', type=str, nargs='+', \
                        default="", help=mes)

    mse = "flag for asvspoof evaluation, since partialspoof use segment label, asvspoof use its protocol."
    parser.add_argument('--data-type', choices=['partialspoof', 'asvspoof'], \
            default='partialspoof', help=mes)


    mse = "This is to save or not for the out of inference: ali (segment score after align), emb."
    parser.add_argument('--save-output', type=str, nargs='+', \
                        default=['ali'], help=mes)



    #
    # done
    if argument_input is not None:
        return parser.parse_args(argument_input)
    else:
        return parser.parse_args()


if __name__ == "__main__":
    pass
    
