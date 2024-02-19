#!/bin/bash
#SBATCH --nodes=1			# Number of requested nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4			# Number of requested cores
#SBATCH --qos=preemptable
#SBATCH --constraint="Tesla"
#SBATCH --out=train_uriel-gram.%j.out		# Output file name
#SBATCH --error=train_uriel-gram.%j.err
#SBATCH --mail-user=enri8153@colorado.edu

# purge all existing modules
module purge

# Load the python module
module load anaconda


# Run Python Script
conda activate bankrank
cd "/projects/enri8153/langrank/"

python3 ./tests/train_file.py -t "MT" 
python3 ./tests/train_file.py -t "MT" -d
