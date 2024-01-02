#!/bin/bash

#Example to measure different types of EER

pred_DIR=$1 	# predicted dir.
metric=$2 	# UttEER, SegEER, RangeEER
dset=$3 	# dev eval

# for RangeEER
#SCALEs="2 4 8 16 32 64 utt"
scale=$4	#Which score branch

#############Utterance-level EER
if [[ ${metric} == "UttEER"  ]]; then
    ASV_SCORES_FILE=${PS_PATH}"/database/protocols/PartialSpoof_LA_asv_scores/PartialSpoof.LA.asv."$dset".gi.trl.scores.txt"
    python ${PS_PATH}/metric/UtteranceEER.py \
        --pred_file ${pred_DIR}/output/log_output_${dset} \
        --asv_score_file ${ASV_SCORES_FILE} \
        --utt2label_file ${PS_PATH}/database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm."$dset".trl.txt
fi


#############Point-based Segment-level EER
if [[ ${metric} == "SegEER"  ]]; then
    RES_DIR=${pred_DIR}/Loc_SegEER/${dset}
    
    python ${PS_PATH}/metric/SegmentEER.py \
        --ref_rttm ${PS_PATH}/database/${dset}/con_data/rttm_2cls_0sil \
        --model_dir ${pred_DIR} \
	--sml_dir exp-01 exp-02 exp-03 \
	--dset ${dset} \
        --reco2dur ${PS_PATH}/database/${dset}/con_data/reco2dur \
	--label2num_file ${PS_PATH}/database/label2num/label2num_2cls_0sil \
        --save_dir ${RES_DIR} 

fi

##############Range-based EER
if [[ ${metric} == "RangeEER"  ]]; then
# Since this processing is quite slow, we provide implement for single score_ali file here.	
    RES_DIR=${pred_DIR}/Loc_RangeEER/${dset}_${scale}
    
    python ${PS_PATH}/metric/RangeEER.py \
        --ref_rttm ${PS_PATH}/database/${dset}/con_data/rttm_2cls_0sil_spoof \
        --hyp_sco_ali ${pred_DIR}/output/${dset}_score_ali_${scale}_*.pkl \
        --reco2dur_file ${PS_PATH}/database/${dset}/con_data/reco2dur \
        --scale ${scale} \
        --save_dir ${RES_DIR} 

fi

