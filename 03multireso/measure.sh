#!/bin/bash

# This script summerize how we measure the perfomrance.


SCALEs="2 4 8 16 32 64 utt"


############## Utterance EER ###############





############## Point-based Segment EER ###############

# For multi-reso
python ../metric/SegmentEER.py \
        --model_dir ./multi-reso \
        --save_dir multi-reso/SegEER \
        --sml_dir exp-01 exp-02 exp-03 


#For single-reso
for scale in ${SCALEs}; do 
    python ../metric/SegmentEER.py \
	    --model_dir single-reso/${scale} \
	    --save_dir single-reso/${scale}/SegEER \
	    --sml_dir exp-01 exp-02 exp-03 \
	    --keyword _${scale}_
#Here we use keyword to only measure performance for the output from the corresponding score branch
done
############## Point-based Segment EER ###############



############## Range-based EER ###############

############## Range-based EER ###############
