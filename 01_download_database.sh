#!/bin/bash
	
FILE_NAMEs=(train dev eval segment_labels_v1.2 protocols)

FILE_NAMEs=(segment_labels_v1.2)


for file in ${FILE_NAMEs}; do

    link="https://zenodo.org/record/5766198/files/database_"${file}".tar.gz?download=1"
    if [ ! -e ./database/${file} ];then
        echo -e "${RED}Downloading PartialSpoof ${name}"
	echo ${link}
        wget -q --show-progress -c ${link} -O  database_${file}.tar.gz
    fi
    tar -zxvf database_${file}.tar.gz
    rm database_${file}.tar.gz
done



