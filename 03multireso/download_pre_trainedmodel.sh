#!/bin/bash

FILE_NAMEs=(multi-reso single-reso)

for file in ${FILE_NAMEs}; do
    link="https://zenodo.org/record/6674660/files/"${file}".tar.gz?download=1"
    if [ ! -e ./${file}/01/trained_network.pt ]; then
        echo -e "${RED}Downloading pretrained models forPartialSpoof"
        echo ${link}
        wget -q --show-progress -c ${link} -O  ${file}.tar.gz
    fi
    tar -zxvf ${file}.tar.gz
    rm ${file}.tar.gz
done

