As shown in the figure, we support three types EERs for detection and localization in the Partial Spoof scenario.

1. Utterance-level EER
2. Segment-level EER
3. [RangeEER](https://arxiv.org/pdf/2305.17739.pdf)

<img src="../Figures/EERs.jpg" />



We recommand to use **Utterance EER** for spoof detection and **RangeEER** for localization.





## Prepare

1. Install [pyannote](https://github.com/pyannote/pyannote-metrics)

```shell
pip install pyannote.metric
pip install pyannote.core
pip install pyannote.database
```



2. Active environment.

```shell
cd PartialSpoof
source ./env.sh
```



## Usage

I included an example in `cal_EER.sh` to help you measure performance. Feel free to create your own as needed.


```shell
bash <Path_to_PartialSpoof>/metric/cal_EER.sh <pred_DIR> UttEER <dset> <scale>
# <pred_DIR> can be set to `.` if running from the same directory (which has output/log_output_*)
# <metric> can be one of UttEER, SegEER, or RangeEER
# <dset> refers to the dataset, either dev or eval
# <scale> is specifically used for RangeEER
# NOTE: For convient, our implementation uses SegmentEER is for multiple seeds, which will iteratively measure perfomrance with different random seeds and average them. Thus please ensure that <pred_DIR> is set to the folder containing all sub-folders (each with different random seeds) for SegEER calculations. 
```
Below are instruction for each type of EER within `cal_EER.sh`



### Utterance EER (Spoof detection)

```shell
bash <Path_to_PartialSpoof>/metric/cal_EER.sh multi-reso/exp-01 UttEER dev
# Input is multi-reso/exp-01/log_output_dev, which has utt scores from all score branshes: [S_1, ..., S_K] as in model.py:573
# Output all EER and threshold vectors
# For example [EER_1, ..., EER_k, ..., EER_K], where EER_m is the k-th score branch.
# default corresponding to measurment use score branch: 20ms, ..., 640ms.

# It will implemented as:
ASV_SCORES_FILE=${PS_PATH}"/database/protocols/PartialSpoof_LA_asv_scores/PartialSpoof.LA.asv."$dset".gi.trl.scores.txt"
python ${PS_PATH}/metric/UtteranceEER.py \
    --pred_file ${pred_DIR}/output/log_output_${dset} \
    --asv_score_file ${ASV_SCORES_FILE}
```

* `--pred_file` point to the predicted file.
* `--asv_score_file` point to the asv score, which is used to calculate min-tdcf.
But I personally do not recommend using t-DCF, because Partial Spoof is designed not only for ASVspoof, which aims to deceive machines, but also for DeepFake, which is intended to fool humans.



### Range-based EER (for spoof localization)

```shell
bash <Path_to_PartialSpoof>/metric/cal_EER.sh multi-reso/exp-01 RangeEER dev 64
# Outputed values will be saved to ${RES_DIR}/ set by RangeEER.py:242, 277
# and will print out those value real-time, each row has below format:
# [[threshold, fpr, fnr, abs(fpr-fnr)]]
# [[... ...]]
# RangeEER=XXX, with threshold = XXX
# 
#-------------------------------------------------------
RES_DIR=${pred_DIR}/Loc_RangeEER/${dset}_${scale}

python ${PS_PATH}/metric/RangeEER.py \
    --ref_rttm ${PS_PATH}/database/${dset}/con_data/rttm_2cls_0sil_spoof \
    --hyp_sco_ali ${pred_DIR}/output/${dset}_score_ali_${scale}_*.pkl \
    --reco2dur_file ${PS_PATH}/database/${dset}/con_data/reco2dur \
    --scale ${scale} \
    --save_dir ${RES_DIR} 
```

* `--ref_rttm` Path to the ground-truth timestamp annotations file in the rttm format.
* `--hyp_sco_ali`Path to the predicted score alignment pkl file (designated as **score_ali**).
* `--reco2dur_file` Path to the recording duration file.
* `--scale` The resolution of the predicted scores, used to map uniform predictions to timestamps.
* `--save_dir` Directory where the calculated results (det curve) will be saved.

More options could be find in the `RangeEER.py`





### Segment EER (for spoof localization)
The `cal_EER.sh` script can be used as follows to calculate SegmentEER for the 'dev' dataset in a multi-resolution model:

```shell
bash <Path_to_PartialSpoof>/metric/cal_EER.sh multi-reso SegEER dev
# Save detailed results into ${pred_DIR}/Loc_SegEER/${dset}.npz
# Output the following:
# loading ${model_dir}/${sub_dir}/output/dev_score_ali_64_xxx.pkl
# ...
# loading ${model_dir}/${sub_dir}/output/dev_score_ali_2_xxx.pkl   
# eer:
# [[...]] 
#
# The above output has the dimension as [Pred_dim x Measure_dim]. 
# The default value for Pred_dim is 6, set by Pred_Resolutions, which corresponds to the output from each score branch (20, 40, 80, 160, 320, 640 ms).
# The default value for Measure_dim is 8, set by Measure_Resolutions + utt, which corresponds to different measuring resolutions (10, 20, 40, 80, 160, 320, 640 ms, and Utt).
#-------------------------------------------------------

RES_DIR=${pred_DIR}/Loc_SegEER/${dset}

python ${PS_PATH}/metric/SegmentEER.py \
    --ref_rttm ${PS_PATH}/database/${dset}/con_data/rttm_2cls_0sil \
    --model_dir ${pred_DIR} \
    --sub_dir exp-01 exp-02 exp-03 \
    --dset ${dset} \
    --reco2dur ${PS_PATH}/database/${dset}/con_data/reco2dur \
    --label2num_file ${PS_PATH}/database/label2num/label2num_2cls_0sil \
    --save_dir ${RES_DIR} 
```

* `--ref_rttm`   Path to the ground-truth timestamp annotations file in the rttm format.
* `--model_dir` Directory containing all sub-folders for SegEER calculations, each representing a different random seed. Since Segment EER is implemented to average EER for results from all random seeds.
* `--sub_dir`  List of sub_dirs for experiments with different random seeds. Set as '.' if model_dir is the sole experiment directory.
* `--dset` Specifies the dataset (e.g., 'dev', 'eval').
* `--reco2dur` Path to the recording duration file.
* `--label2num_file` For converting textual labels in ground-truth files (like 'spoof'/'bonafide') to numerical labels. Default mapping is: {'spoof': 0, 'bonafide': 1}.
* `--save_dir ` Directory where the calculated results will be saved.

More options could be find in the `SegmentEER.py`







## Note

For localization tasks, we offer two types of metrics: **Segment-level EER** (SegmentEER) for point-based evaluation, and **Range-based EER** (RangeEER).

> For fair and proper evaluation of spoof localization models, we recommend using range-based EER, or point-based EER that uses references with a finer temporal resolution than that used during model training. In addition, when the training temporal resolution is unknown, the range-based EER would be a more appropriate choice.

For more details, refer to the paper:  [Range-Based Equal Error Rate for Spoof Localization](https://arxiv.org/abs/2305.17739) . For quick compare and understand SegmentEER and RangeEER, please see below figure (in the poster of [RangeEER paper](https://github.com/nii-yamagishilab/PartialSpoof/blob/main/Figures/PartialSpoof_rangeEER_IS23poster.pdf)).

<img src="../Figures/SegmentEER_vs_RangeEER.jpg" />

:exclamation:If you have to choose one metric, **I recommend using range-based EER**. This approach facilitates fairer comparisons of RangeEER across different studies. 

But given the following points: (1) The current implementation of RangeEER is quite slow. (2) SegmentEER results at a resolution finer than the training resolution (such as 10ms) are similar to RangeEER, as indicated in Tables 3 and 4 of the paper [Range-Based Equal Error Rate for Spoof Localization](https://arxiv.org/abs/2305.17739). For immediate results, consider using SegmentEER at a finer resolution (like 10ms) as a temporary measure of performance. You can then wait for the more precise values calculated by RangeEER.





## Citation

```latex
@inproceedings{zhang23v_interspeech,
  author={Lin Zhang and Xin Wang and Erica Cooper and Nicholas Evans and Junichi Yamagishi},
  title={{Range-Based Equal Error Rate for Spoof Localization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={3212--3216},
  doi={10.21437/Interspeech.2023-1214}
}
```



The current implementation of RangeEER is based on pyannote.metric. I must offer my apologies here. While revising our paper for the double-blind review process, I temporarily removed the citation for pyannote.metric but subsequently forgot to add it back. I sincerely apologize to the authors of pyannote.metric for this oversight. If you are utilizing RangeEER in your work, I strongly encourage you to cite the significant contributions of pyannote.metric. Thank you!

```latex
@inproceedings{pyannote.metrics,
  author = {Herv\'e Bredin},
  title = {{pyannote.metrics: a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems}},
  booktitle = {{Interspeech 2017, 18th Annual Conference of the International Speech Communication Association}},
  year = {2017},
  month = {August},
  address = {Stockholm, Sweden},
  url = {http://pyannote.github.io/pyannote-metrics},
}
```



## Dependent Repositories

* [pyannote.metrics](https://github.com/pyannote/pyannote-metrics)



## License

This repo is mainly licensed under the BSD 3-Clause License (`PartialSpoof/LICENSE`). 

