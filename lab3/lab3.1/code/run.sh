#!/bin/bash

conda activate textlab
jupyter nbconvert --to notebook --execute --inplace explore.ipynb
jupyter nbconvert --to notebook --execute --inplace part_1.ipynb
jupyter nbconvert --to notebook --execute --inplace part_2_bow.ipynb
jupyter nbconvert --to notebook --execute --inplace part_2_w2v.ipynb
jupyter nbconvert --to notebook --execute --inplace part_2_glv.ipynb