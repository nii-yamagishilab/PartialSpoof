#!/bin/bash

set -x
CON_PATH=../../../database
OUTPUT_DIR=output
time_tag=`date +%Y%m%d`
if [ ! -d ${OUTPUT_DIR}  ]; then
	mkdir ${OUTPUT_DIR}
fi


python -m pdb  main.py --inference --module-model model --model-forward-with-file-name --module-config config_ps.config_test_on_dev  \
       --temp-flag ${CON_PATH}/segment_labels/dev_seglab_0.01.npy \
       --output-dir ${OUTPUT_DIR}/dev #> ${OUTPUT_DIR}/log_output_dev 2>&1 & 
##
#python main.py --inference --module-model model --model-forward-with-file-name  --module-config config_con.config_con_test_on_eval\
#       --temp-flag ${CON_PATH}/segment_labels/eval_seglab_0.01.npy \
#       --output-dir ${OUTPUT_DIR}/eval > ${OUTPUT_DIR}/log_output_eval 2>&1  
