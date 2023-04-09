#!/bin/bash
set -x

# 1. Download SSL models
if [ ! -e ../modules/ssl_pretrain ]; then
    mkdir -p ../modules/ssl_pretrain	
fi

ssl_model="w2v_large_lv_fsh_swbd_cv.pt"
if [ ! -e ../modules/ssl_pretrain/${ssl_model} ]; then
    wget --show-progress https://dl.fbaipublicfiles.com/fairseq/wav2vec/${ssl_model} ${ssl_pretrain}/ 	
fi    	

# 2. Download Lin Zhang's pretrained models.
FILE_NAMEs=(multi-reso single-reso)

for file in ${FILE_NAMEs}; do
    link="https://zenodo.org/record/6674660/files/"${file}".tar.gz?download=1"
    if [ ! -e ./${file}/01/trained_network.pt ]; then
        echo -e "${RED}Downloading pretrained models for PartialSpoof"
        echo ${link}
        wget -q --show-progress -c ${link} -O  ${file}.tar.gz
    fi
    tar -zxvf ${file}.tar.gz
    rm ${file}.tar.gz
done

