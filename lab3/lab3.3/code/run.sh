#!/bin/bash

# (usage) sbatch run.sh

#SBATCH --account=mth240012p       # don't change this
#SBATCH --job-name=part_1.ipynb
#SBATCH --cpus-per-task=1         # GPU-shared allows max 5 cpus per GPU
#SBATCH --time 1:00:00

#SBATCH -o test_ipynb.out          # write job console output to file test.out
#SBATCH -e test_ipynb.err          # write job console errors to file test.err
#SBATCH --mem=60G
#SBATCH --partition=GPU-shared     # don't change this unless you need 8 GPUs
#SBATCH --gpus=v100-32:1             # don't increase this unless you need more than 1 GPU
# request a 16GB or 32GB GPU)

module load anaconda3
conda activate textlab
jupyter nbconvert --to notebook --execute --inplace part3_1.ipynb
jupyter nbconvert --to notebook --execute --inplace lora_finetuning.ipynb
jupyter nbconvert --to notebook --execute --inplace part3_1_lora.ipynb
jupyter nbconvert --to notebook --execute --inplace part3_2.ipynb
jupyter nbconvert --to notebook --execute --inplace part3_3.ipynb
echo "Successfully ran files"