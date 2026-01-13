#!/bin/bash
conda activate stat214 
python map_data.py
python clean.py
python model.py 
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb