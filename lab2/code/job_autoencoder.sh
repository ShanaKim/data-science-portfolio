#!/bin/bash

# (usage) sbatch job_autoencoder.sh configs/default.yaml

# our script starts with a bunch of Slurm options
#SBATCH --account=mth240012p       # don't change this
#SBATCH --job-name=ae-test
#SBATCH --cpus-per-task=5          # GPU-shared allows max 5 cpus per GPU
#SBATCH --time 10:00:00
#SBATCH -o deeper_ae.out          # write job console output to file test.out
#SBATCH -e deeper_ae.err          # write job console errors to file test.err

#SBATCH --partition=GPU-shared     # don't change this unless you need 8 GPUs
#SBATCH --gpus=v100-32:1             # don't increase this unless you need more than 1 GPU
# (can instead specify --gpus=v100-16:1 or --gpus=v100-32:1 to specifically
# request a 16GB or 32GB GPU)

module load anaconda3
conda activate env_214
python run_autoencoder.py $1