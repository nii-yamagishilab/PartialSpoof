#!/bin/bash

#set -x
PS_PATH=/home/smg/zhanglin/workspace/PROJ/Public/CODE/PartialSpoof

pred_DIR=$1
METRIC=$2 #UttEER, SegEER, RangeEER
dset=$3
SCALE=$4 #2 4 8 16 32 64 utt

#Utterance-level EER
if [[ ${METRIC} == "UttEER"  ]]; then
    ASV_SCORES_FILE=${PS_PATH}"/database/protocols/PartialSpoof_LA_asv_scores/PartialSpoof.LA.asv."$dset".gi.trl.scores.txt"
    python ${PS_PATH}/metric/UtteranceEER.py \
        --pred_file ${pred_DIR}/output/log_output_${dset} \
        --asv_score_file ${ASV_SCORES_FILE}
fi

#Range-based EER
if [[ ${METRIC} == "RangeEER"  ]]; then
    RES_DIR=Loc_RangeEER/${dset}_${SCALE}
    
    if [[ ! -d ${RES_DIR} ]]; then
    	mkdir -p ${RES_DIR}
    fi

    python -u ${PS_PATH}/metric/RangeEER.py \
        --ref_rttm ${PS_PATH}/database/${dset}/con_data/rttm_2cls_0sil_spoof \
        --hyp_sco_ali ${pred_DIR}/output/${dset}_score_ali_${SCALE}_*.pkl \
        --reco2dur ${PS_PATH}/database/${dset}/con_data/reco2dur \
        --scale ${SCALE} \
        --save_dir ${RES_DIR} 

fi

