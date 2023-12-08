<div align=center>
<img src="Figures/PartialSpoof_logo.png" width="60%">
</div>
Welcome to the story of PartialSpoof! This is the implementation of papers for Partial Spoof. And this repo is adapted from [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts). Here are some links you might interested:

* [:arrow_down: PartialSpoof Database](https://zenodo.org/record/5766198)
* [:headphones: Sample](https://nii-yamagishilab.github.io/zlin-demo/IS2021/index.html)
* [:woman_technologist: Github for model](https://github.com/nii-yamagishilab/PartialSpoof) (You are here!).  :point_left::point_left:
* [:woman_technologist: Github for data construction  (TBA)](https://github.com/nii-yamagishilab/PartialSpoof_database)

* :page_with_curl:Papers: Please refer to the link in [Folder and its paper](#folderintro)



Please feel free to give suggestions and feedback. : )

Lin Zhang; Xin Wang; Erica Cooper; Nicholas Evans; Junichi Yamagishi



# Outline

0. [Updation](#update)
2. [Folder and Its Paper](#folderintro)
3. [Folder Structure](#folderstructure)
4. [Citation](#citation)
5. [Acknowledgments and License](#ack)

# <a name="update"/> 0. Updation
* 2023-12: add metrics: EER for spoof detection; SegmentEER and RangeEER for spoof localization.
* 2023-12: add folders for multiple random seeds, and update readme.
* 2022-12: release multi-reso. and single-reso CMs.




# <a name="folderintro"/> 1. Folder and its paper

| Folder         | Paper                                                        |
| -------------- | ------------------------------------------------------------ |
| 00data-prepare | Processing to generate PartialSpoof database and automatic annotation. (To be released) |
| 01singletask   | CM trained on the single task (either utterance-level or segment-level detection) in the paper [An Initial Investigation for Detecting Partially Spoofed Audio](https://nii-yamagishilab.github.io/publication/zhang-21-ca-interspeech/) (To be released) |
| 02multitask    | CM trained on multi tasks (both utterance-level and segment-level detection) in the paper [Multi-task Learning in Utterance-level and Segmental-level Spoof Detection](https://nii-yamagishilab.github.io/publication/zhang-21-asvspoof/) (To be released) |
| 03multireso    | Multi resolution CM in the paper [The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance](https://ieeexplore.ieee.org/document/10003971) |
| metric         | metric used for spoof (utterance-level) detection and (segment-level) localization [Range-Based Equal Error Rate for Spoof Localization](https://arxiv.org/abs/2305.17739) ~~(To be released)~~ |



Please go to the `[Folder]/README.md` to read details of usages.



# <a name="folderstructure"/> 2. Folder structure

```
PartialSpoof
├── 01_download_database.sh			: Script used to download PartialSpoof from zenodo.
├── 03multireso
│   ├── 01_download_pretrained_models.sh	: Script used to download pretrained models.
│   ├── main.py
│   ├── model.py			: Model structure and loss are in here! same for multi/single-reso.
│   ├── multi-reso		: folder for multi-reso model
│   ├── README.md
│   └── single-reso		: folder for single-reso model
│       └── {2, 4, 8, 16, 32, 64, utt}
├── config_ps				: Config files for experiments
│   ├── config_test_on_dev.py
│   └── config_test_on_eval.py
├── env.sh						
├── Figures
│   ├── EERs.pdf
│   └── PartialSpoof_logo.png
├── LICENSE
├── metric			
│   ├── cal_EER.sh
│   ├── RangeEER.py
│   ├── README.md
│   ├── rttm_tool.py
│   ├── SegmentEER.py
│   └── UtteranceEER.py
├── database					: PartialSpoof Databases
│   ├── train
│   ├── dev						: Folder for dev set
│   │   ├── con_data	: related data file. (following kaldi format)
│   │   ├── con_wav		: waveform
│   │   └── dev.lst		: waveform list
│   ├── eval
│   ├── label2num			: convert string labels to numerical labels.
│   │   └── label2num_2cls_0sil		: bonafide/spoof (More to be released)
│   ├── protocols
│   ├── segment_labels
│   └── vad
│       ├── dev
│       ├── eval
│       └── train
├── modules
│   ├── gmlp.py
│   ├── LICENSE
│   ├── multi_scale
│   │   └── post.py
│   ├── s3prl  	     			: s3prl repo 
│   └── ssl_pretrain 			: Folder to save downloaded pretrained ssl model
├── project-NN-Pytorch-scripts.202102	: Modified project-NN-Pytorch-scripts repo
└── README.md

```



# <a name="citation"/> 3. Citation

It is appreciated if you can cite the corresponding paper when the idea, code, and pretrained model are helpful to your research.

```
@inproceedings{zhang21ca_interspeech,
  author={Lin Zhang and Xin Wang and Erica Cooper and Junichi Yamagishi and Jose Patino and Nicholas Evans},
  title={{An Initial Investigation for Detecting Partially Spoofed Audio}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={4264--4268},
  doi={10.21437/Interspeech.2021-738}
}

```

```
@inproceedings{zhang21_asvspoof,
  author={Lin Zhang and Xin Wang and Erica Cooper and Junichi Yamagishi},
  title={{Multi-task Learning in Utterance-level and Segmental-level Spoof Detection}},
  year=2021,
  booktitle={Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge},
  pages={9--15},
  doi={10.21437/ASVSPOOF.2021-2}
}
```

```
@article{10003971,
  author={Zhang, Lin and Wang, Xin and Cooper, Erica and Evans, Nicholas and Yamagishi, Junichi},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance}, 
  year={2023},
  volume={31},
  number={},
  pages={813-825},
  doi={10.1109/TASLP.2022.3233236}}
```



# <a name="ack"/>4. Acknowledgments

This study is partially supported by the Japanese-French joint national VoicePersonae project supported by JST CREST (JPMJCR18A6, JPMJCR20D3), JPMJFS2136 and the ANR (ANR-18-JSTS-0001), MEXT KAKENHI Grants (21K17775, 21H04906, 21K11951, 18H04112), Japan, and Google AI for Japan program.



# License

This project is mainly licensed under the BSD 3-Clause License (`./LICENSE`). 
Each folder within the project may contain their corresponding LICENSE according to the external libraries used. Please refer to the README.md file in each folder for more details. 

Additionally, specific licenses for some of the external libraries used are mentioned below:
* `modules/s3prl` is licensed under the MIT License (`modules/s3prl/LICENSE.txt`), but please note that the latest version of s3prl is now under the Apache License version 2.0. 
* `project-NN-Pytorch-scripts.202102` is licensed under the BSD 3-Clause License (`project-NN-Pytorch-scripts.202102/LICENSE`). 
* `modules/gmlp.py`  is licensed under the MIT License (`modules/LICENSE`)
