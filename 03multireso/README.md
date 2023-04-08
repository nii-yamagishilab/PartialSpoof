# PartialSpoof/03multireso

This is the implementation for paper:

 [The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance](https://ieeexplore.ieee.org/document/10003971)



Please cite database and this paper if you use:
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



## Prepare

* Environment
  SSL model used in this project is based on s3prl. Please following [s3rpl#installation](https://github.com/s3prl/s3prl#installation) to build the enviorment. Here are two ways for downloading:
  
  * Option 1. If you have downloaded s3prl, Please link it to `modules/s3prl`
  
  `ln -s <path_to_s3prl> ../modules/s3prl`
  
  * Option 2. Otherwise, please download s3prl through submodule. 
  
  ```shell
  cd .. #cd to PartialSpoof, since it is registered for path './'
  git submodule init
  git submodule update
  ```
  
  


* Database and label: PartialSpoof

  Please download database from zenodo if you don't have:

  `bash 01_download_database.sh`



* Pre-trained models 

  If you want to use the pretrained model, please download them by:

  `bash 01_download_model.sh 1 `



## Usuage

1. env
   * modify the `CON_DATA_PATH` in the line 27 of config\*py to your path.
   
     ```
     sed -i config_ps/config*py 's/\/path\/to\/data/<Your_path>/g'
     ```
   
     
   
   * active path
   
     ```shell
     $ cd 03multireso
     $ conda activate ssl
     $ source ./env.sh
     $ cd multi-reso #cd one folder
     ```
   



3. Run `bash run.sh`, which including training and inference.

   ```shell
   $ bash 00_run.sh
   ```

   `00_run.sh` includes three stages:

   

   * *stage = 0: Check and download ssl model.*

   
   
   * *stage = 1: Training*
   
   ```shell
   CON_PATH=/path/of/database
   OUTPUT_DIR=output #/dir/to/save/output
   python main.py --module-model model --model-forward-with-file-name --seed 1 \
           --ssl-finetune \
           --multi-scale-active utt 64 32 16 8 4 2 \
           --num-workers 4 --epochs 5000 --no-best-epochs 50 --batch-size 8 --not-save-each-epoch\
           --sampler block_shuffle_by_length --lr-decay-factor 0.5 --lr-scheduler-type 1 --lr 0.00001 \
           --module-config config_ps.config_test_on_eval \
           --temp-flag ${CON_PATH}/segment_labels/train_seglab_0.01.npy \
           --temp-flag-dev ${CON_PATH}/segment_labels/dev_seglab_0.01.npy --dev-flag >  ${OUTPUT_DIR}/log_train 2> ${OUTPUT_DIR}/log_err
   ```
   
   This stage will generate below files:
   
   >PS\_{trn, dev}\_utt_length.dic: file to save mean / std of the input.
   >
   >output/log_train:  log file for training.
   >
   >trained_network.pt: trained model.
   >
   
   
   
   * *Stgae = 3: Inference*
   
   ```shell
     python main.py --inference --module-model model_debug_con --model-forward-with-file-name --module-config config_ps.config_test_on_dev  \
            --temp-flag ${CON_PATH}/segment_labels/dev_seglab_0.01.npy \
            --output-dir ${OUTPUT_DIR}/dev > ${OUTPUT_DIR}/log_output_dev 2>&1 & 
   
   
     python main.py --inference --module-model model_debug_con --model-forward-with-file-name  --module-config config_ps.config_test_on_eval\
            --temp-flag ${CON_PATH}/segment_labels/eval_seglab_0.01.npy \
            --output-dir ${OUTPUT_DIR}/eval > ${OUTPUT_DIR}/log_output_eval 2>&1  &
   ```
   
   This stage will generate below files:
   
   >output/
   >
   >\*_emb\*.pkl: saved embedding.
   >
   >\*_score_ali\*.pkl: saved segment-level score in each sacle.
   >
   >log_output_\*: log file for inference.
   
   



### Notes

`multi-reso` and `single-reso` are almost the same, the only difference is option `--multi-scale-active [resolution] `  in the 00_run.sh. 

For example:

* ` --multi-scale-active utt 64 32 16 8 4 2 ` refers to use all scales to train the model.

* ` --multi-scale-active 16  ` refers to use resolution=160 ms to train the model.

To get detail for each option, please go to `project-NN-Pytorch-scripts.202102/core_scripts/config_parse/arg_parse.py`



## Reference Repositories

* [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts)
* [s3prl](https://github.com/s3prl/s3prl)
* [gmlp](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/aacd926dba97ba7a1d67a3631120c46d0670ba94/labml_nn/transformers/gmlp/__init__.py)





## LICENSE
03multireso project is mainly licensed under the BSD 3-Clause License (`PartialSpoof/LICENSE`). 

External libraries and their corresponding licenses in this project are listed below:

* `modules/s3prl` is licensed under the MIT License (`modules/s3prl/LICENSE.txt`), but please note that the latest version of s3prl is now under the Apache License version 2.0. 
* `project-NN-Pytorch-scripts.202102` is licensed under the BSD 3-Clause License (`project-NN-Pytorch-scripts.202102/LICENSE`). 
* `modules/gmlp.py` file is licensed under the MIT License (`modules/LICENSE`)

