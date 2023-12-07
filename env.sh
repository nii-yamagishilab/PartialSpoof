#!/bin/bash
# if necessary, load conda environment
ROOT_PATH=`pwd`
XW_PATH=${ROOT_PATH}/../project-NN-Pytorch-scripts.202102
MODULES_PATH=${ROOT_PATH}/../modules

S3PRL_PATH=${MODULES_PATH}/s3prl
export PYTHONPATH=${XW_PATH}:${ROOT_PATH}/..:$PYTHONPATH:$MODULES_PATH:${S3PRL_PATH}
export PS_PATH=$ROOT_PATH

