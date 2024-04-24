#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --qos=preemptable
#SBATCH --constraint=rtx6000|A100|A40|V100
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --error=ud_logs/test.%jerr
#SBATCH --out=ud_logs/test.%j.out

source /curc/sw/anaconda3/latest
conda activate bankrank


Rscript train_pos_model.R -c $1 -m $2