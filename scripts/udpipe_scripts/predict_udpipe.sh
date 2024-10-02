#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --qos=preemptable
#SBATCH --constraint=rtx6000|A100|A40|V100
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --error=ud_logs/weird_preds.%j.err
#SBATCH --out=ud_logs/weird_preds.%j.out


words=$LANGUAGES
read -ra words <<< "$LANGUAGES"

tgt="${words[0]}"
transfer="${words[1]}"

echo "$tgt"
echo "$transfer"

source /curc/sw/anaconda3/latest
conda activate udpipe

Rscript ./scripts/udpipe_scripts/udpipe_predict.R -t "$tgt" -r "$transfer"