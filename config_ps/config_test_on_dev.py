#!/usr/bin/env python
"""
config.py for PartialSpoof 

Usage: 
 For training, change Configuration for training stage
 For inference,  change Configuration for inference stage
"""

__author__ = "Lin Zhang"
__email__ = "zhanglin@nii.ac.jp"
__copyright__ = "Copyright 2021, Lin Zhang"

#########################################################
## Configuration for training stage
#########################################################

exp_path='exp/'
# Name of datasets
#  after data preparation, trn/val_set_name are used to save statistics 
#  about the data sets
trn_set_name = 'PS_trn'
val_set_name = 'PS_dev'

# for convenience

#CON_DATA_PATH = '/home/smg/zhanglin/workspace/PROJ/Public/PartialSpoof/database/'
CON_DATA_PATH = '/path/to/data'

# File lists (text file, one data name per line, without name extension)
# trin_file_list: list of files for training set
trn_list = CON_DATA_PATH + '/train/train.lst'  
# val_file_list: list of files for validation set. It can be None
val_list = CON_DATA_PATH + '/dev/dev.lst'  

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  fro con, train and validation data are not in the same sub-directory
trn_input_dirs = [CON_DATA_PATH + '/train/con_wav/']
val_input_dirs = [CON_DATA_PATH + '/dev/con_wav/']

# Dimensions of input features
# input_dims = [dimension_of_feature_1, dimension_of_feature_2, ...]
input_dims = [1]

# File name extension for input features
# input_exts = [name_extention_of_feature_1, ...]
# Please put ".f0" as the last feature
input_exts = ['.wav']

# Temporal resolution for input features
# input_reso = [reso_feature_1, reso_feature_2, ...]
#  for waveform modeling, temporal resolution of input acoustic features
#  may be = waveform_sampling_rate * frame_shift_of_acoustic_features
#  for example, 80 = 16000 Hz * 5 ms 
input_reso = [1]

# Whether input features should be z-normalized
# input_norm = [normalize_feature_1, normalize_feature_2]
input_norm = [False]
    
# Similar configurations for output features
output_dirs = []
output_dims = [1]
output_exts = ['.bin']
output_reso = [1]
output_norm = [False]

# Waveform sampling rate
#  wav_samp_rate can be None if no waveform data is used
wav_samp_rate = 16000

# Truncating input sequences so that the maximum length = truncate_seq
#  When truncate_seq is larger, more GPU mem required
# If you don't want truncating, please truncate_seq = None
truncate_seq = None

# Minimum sequence length
#  If sequence length < minimum_len, this sequence is not used for training
#  minimum_len can be None
minimum_len = None
    

# Optional argument
#  Just a buffer for convenience
#  It can contain anything
#optional_argument = ['/data/protocol.txt']

#########################################################
## Configuration for inference stage of test
#########################################################
# similar options to training stage

set_type='dev'
test_input_path = CON_DATA_PATH + set_type
test_set_name = 'PS_' + set_type 
#test_minimum_len = 160*16 #already modified in model_debug.py

# List of test set data
# for convenience, you may directly load test_set list here
#test_list = tmp + '/scp/test.lst'
test_list = test_input_path + '/'+set_type+'.lst'

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  we assume train and validation data are put in the same sub-directory
test_input_dirs = [test_input_path + '/con_wav/']

# Directories for output features, which are []
test_output_dirs = []

