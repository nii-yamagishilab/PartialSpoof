#!/bin/bash
	
FILE_NAMEs=(single-reso multi-reso)

for file in ${FILE_NAMEs}; do

    link="https://zenodo.org/record/6674660/files/"${file}".tar.gz?download=1"
    if [ ! -e ./database/${file} ];then
        echo -e "${RED}Downloading PartialSpoof ${name}"
	echo ${link}
        wget -q --show-progress -c ${link} -O  database_${file}.tar.gz
    fi
    tar -zxvf database_${file}.tar.gz
    rm database_${file}.tar.gz
done



