#!/bin/bash

#SBATCH --mail-user=enri8153@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --qos=preemptable
#SBATCH --constraint=rtx6000|A100|A40|V100
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --error=logs/new_ncdg.%j.err
#SBATCH --out=logs/new_ncdg.%j.out

source /curc/sw/anaconda3/latest
conda activate bankrank

COMBOS=$(cat "$COMBOS_FILE")

python3 /projects/enri8153/langrank/tests/run_one.py "${COMBOS}"
