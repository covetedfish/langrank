#!/bin/bash

#SBATCH --mail-user=enri8153@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
##SBATCH --partition=blanca-kann
#SBATCH --qos=preemptable
##SBATCH --constraint=rtx6000|A100|A40|V100
##SBATCH --qos=blanca-curc-gpu
##SBATCH --partition=blanca-curc-gpu
##SBATCH --account=blanca-curc-gpu
##SBATCH --partition=aa100
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --error=logs/run_all_5.%j.err
#SBATCH --out=logs/run_all_5.%j.out

source /curc/sw/anaconda3/latest
conda activate bankrank

COMBOS=$(cat "$COMBOS_FILE")
# COMBOS=$(cat "$1")

python3 /projects/enri8153/langrank/tests/run_one.py "${COMBOS}"
