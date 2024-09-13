#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=preemptable
#SBATCH --constraint=rtx6000|A100|A40|V100
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --error=ud_default_logs/comparison.%j.err
#SBATCH --out=ud_default_logs/comparison.%j.out

source /curc/sw/anaconda3/latest
conda activate udpipe

Rscript ./scripts/udpipe_scripts/udpipe_train.R -t "ind" -r "swe"