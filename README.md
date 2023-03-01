# PartialSpoof 

This is the implementation of papers for Partial Spoof Project. And this code is adapted from [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts)
Please feel free to give suggestions and feedback. :)

Lin Zhang; Xin Wang; Erica Cooper; Nicholas Evans; Junichi Yamagishi



## Folder and its paper

| Folder      | Paper                                                        |
| ----------- | ------------------------------------------------------------ |
| 03multireso | Multi resolution CM in the paper [The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance](https://ieeexplore.ieee.org/document/10003971) |



Please go to the `[Folder]/README.md` to read details of usages.

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



## Folder structure

```
PartialSpoof
├── 01_download_database.sh
├── 03multireso
│   ├── env.sh
│   ├── main.py
│   ├── model.py
│   ├── multi-reso
│   └── single-reso
│       └── {2, 4, 8, 16, 32, 64, utt}
├── config_ps
│   ├── config_test_on_dev.py
│   └── config_test_on_eval.py
├── database
│   ├── dev
│   │   ├── con_data
│   │   └── con_wav
│   ├── eval
│   ├── segment_labels
│   └── train
├── modules
│   ├── gmlp.py
│   ├── multi_scale
│   │   └── post.py
│   ├── s3prl 
│   └── ssl_pretrain 
├── project-NN-Pytorch-scripts.202102
└── README.md
```



## Notes

This project is adapted from [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts), and each script in the folder has the same function.



## Acknowledgments

This study is partially supported by the Japanese-French joint national VoicePersonae project supported by JST CREST (JPMJCR18A6, JPMJCR20D3), JPMJFS2136 and the ANR (ANR-18-JSTS-0001), MEXT KAKENHI Grants (21K17775, 21H04906, 21K11951, 18H04112), Japan, and Google AI for Japan program.




## License

This project is mainly licensed under BSD 3-Clause License (`project-NN-Pytorch-scripts/LICENSE`), following  [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts). And each folder has its corresponding license, please go to LICENSE under each folder to get more detail. 
