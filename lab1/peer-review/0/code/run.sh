#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh

if ! conda env list | grep -q "stat214"; then
    conda env create -n stat214 -f environment.yaml
fi

conda activate stat214

jupyter nbconvert --to notebook --execute --inplace Lab1_Explanatory_Data_Analysis.ipynb

conda deactivate
