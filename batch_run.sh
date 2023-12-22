#!/bin/bash
set -e

# make sure to activate the conda environment
# conda activate "env name"


echo "Running experimental models"
python train.py --config ./configs/experimental.json
