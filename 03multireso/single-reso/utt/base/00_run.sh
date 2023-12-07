#!/bin/bash

stage=$1
CON_PATH=../../../database
OUTPUT_DIR=output
if [ ! -d ${OUTPUT_DIR}  ]; then
    mkdir ${OUTPUT_DIR}
fi

#stage 0:
if [ $stage -le 0 ]; then
    ssl_link="https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv.pt"
    if [ ! -f ../../../modules/ssl_pretrain/w2v_large_lv_fsh_swbd_cv.pt  ]; then
        wget -q --show-progress -c ${ssl_link} -O ../../../modules/ssl_pretrain
    fi
fi




#stage 1:
if [ $stage -le 1 ]; then
    python main.py --module-model model --model-forward-with-file-name --seed 1 \
	--ssl-finetune \
	--multi-scale-active utt \
	--num-workers 4 --epochs 5000 --no-best-epochs 50 --batch-size 8 --not-save-each-epoch\
       	--sampler block_shuffle_by_length --lr-decay-factor 0.5 --lr-scheduler-type 1 --lr 0.00001 \
	--module-config config_ps.config_test_on_eval \
	--temp-flag ${CON_PATH}/segment_labels/train_seglab_0.01.npy \
	--temp-flag-dev ${CON_PATH}/segment_labels/dev_seglab_0.01.npy --dev-flag >  ${OUTPUT_DIR}/log_train 2> ${OUTPUT_DIR}/log_err
fi

#stage 2
if [ $stage -le 2 ]; then
    python main.py --inference --module-model model --model-forward-with-file-name --module-config config_ps.config_test_on_dev  \
       --temp-flag ${CON_PATH}/segment_labels/dev_seglab_0.01.npy \
       --output-dir ${OUTPUT_DIR}/dev > ${OUTPUT_DIR}/log_output_dev 2>&1 & 

    python main.py --inference --module-model model --model-forward-with-file-name  --module-config config_ps.config_test_on_eval\
       --temp-flag ${CON_PATH}/segment_labels/eval_seglab_0.01.npy \
       --output-dir ${OUTPUT_DIR}/eval > ${OUTPUT_DIR}/log_output_eval 2>&1  
fi


