#!/bin/bash
if [ ! -d "ret_one" ]; then
    mkdir -p "ret_one"
fi 

sbatch --job-name=evol-train -o "ret_one/%j.out" -e "ret_one/%j.err" evol-train.slurm