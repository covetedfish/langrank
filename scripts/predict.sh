#!/bin/bash
#SBATCH --nodes=1			# Number of requested nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4			# Number of requested cores
#SBATCH --qos=preemptable
#SBATCH --constraint="Tesla"
#SBATCH --out=logs/predict_gram.%j.out		# Output file name
#SBATCH --error=logs/predict_gram.%j.err

# purge all existing modules
module purge

# Load the python module
module load anaconda


# Run Python Script
conda activate bankrank
cd "/projects/enri8153/langrank/"


python3 ./tests/predict_all.py -t "MT" -g
python3 ./tests/predict_all.py -t "DEP" -g


