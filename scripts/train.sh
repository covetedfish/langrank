#!/bin/bash
#SBATCH --nodes=1			# Number of requested nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4			# Number of requested cores
#SBATCH --qos=preemptable
#SBATCH --constraint="Tesla"
#SBATCH --out=logs/train_gram_ablations.%j.out		# Output file name
#SBATCH --error=logs/train_gram_ablations.%j.err

# purge all existing modules
module purge

# Load the python module
module load anaconda


# Run Python Script
conda activate bankrank
cd "/projects/enri8153/langrank/"


python3 ./tests/train_file.py -t "MT" -g
python3 ./tests/train_file.py -t "MT"
python3 ./tests/train_file.py -t "DEP" -g
python ./tests/train_file.py -t "DEP"
