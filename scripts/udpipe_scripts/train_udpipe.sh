#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=preemptable
#SBATCH --constraint=rtx6000|A100|A40|V100
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --error=2000_logs/40_its_test.%j.err
#SBATCH --out=2000_logs/40_its_test.%j.out


words=$LANGUAGES
read -ra words <<< "$LANGUAGES"

tgt="${words[0]}"
transfer="${words[1]}"
echo "$tgt"
echo "$transfer"

source /curc/sw/anaconda3/latest
conda activate udpipe

Rscript ./scripts/udpipe_scripts/udpipe_train.R -t "$tgt" -r "$transfer"