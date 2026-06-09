#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate langchain
nohup python3 -u scripts/score_weak_rms.py > scripts/score_weak_rms.log 2>&1 &
echo $!
