As shown in the figure, we support three types EERs for detection and localization in the Partial Spoof scenario.

1. Utterance-level EER 
2. Segment-level EER
3. RangeEER

<img src="../Figures/EERs.pdf" />



We recommand to use utterance EER for spoof detection and RangeEER for localization.

Utterance-level EER and Segment-level EER are adapted from 

RangeEER is adapted from pyannote





## Prepare

install [pyannote](https://github.com/pyannote/pyannote-metrics)

```shell
pip install pyannote.metric
pip install pyannote.core
pip install pyannote.database
```



## Usage

```shell
#1. cd to the model you want to measure, which includes output folder.
cd single-reso/64/01/

#2. run 
bash <Path_to_PartialSpoof>/metric/cal_EER.sh <pred_DIR> <METRIC> <dset> <SCALE> 
# E.g. bash <Path_to_PartialSpoof>/metric/cal_EER.sh . RangeEER dev 64 
# bash <Path_to_PartialSpoof>/metric/cal_EER.sh . UttEER dev 

# pred_DIR could be `.` if you run this within the folder
# METRIC could be UttEER, SegEER, RangeEER

```