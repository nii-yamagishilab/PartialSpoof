#!/bin/bash

set -e
set -x

base_name=exp-
base_dir=base
seed_pow=$1
#SEEDs="1 2 3"
#for seed_pow in $SEEDs;do
    seed=$((10**(${seed_pow}-1)))
    name=${base_name}${seed_pow}
    rm -rf ${name}
    cp -r ${base_dir} ${name}
    cd ${name}
        sed -i "s/--seed 1 /--seed ${seed} /g" 00_run.sh   #update randome seed in the run.sh
        bash 00_run.sh 0
    cd ..

#done
