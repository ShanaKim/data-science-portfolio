#!/bin/bash

# sbatch get_embed_unlabeled.sh configs/default.yaml checkpoints/deeper_emb32-epoch=049-val_loss=0.0551.ckpt

#SBATCH --account=mth240012p       # don't change this
#SBATCH --job-name=get_embed-test
#SBATCH --cpus-per-task=5          # GPU-shared allows max 5 cpus per GPU
#SBATCH --time 10:00:00
#SBATCH -o test1.out          # write job console output to file test.out
#SBATCH -e test1.err          # write job console errors to file test.err

#SBATCH --partition=GPU-shared     # don't change this unless you need 8 GPUs
#SBATCH --gpus=1             # don't increase this unless you need more than 1 GPU


module load anaconda3
conda activate env_214
echo "The python executable in this environment is:"
which python
python get_embedding_unlabeled.py $1 $2 