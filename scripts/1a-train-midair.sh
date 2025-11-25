#!/bin/bash

savepath=$1;

# FIX: Point --records ONLY to the valid "Kite_training" folder
python main.py --mode=train --dataset="midair" --seq_len=2 --db_seq_len=4 --arch_depth=4 --ckpt_dir="$savepath" --log_dir="$savepath/summaries" --records=data/midair/train_data/Kite_training --path="/home/tamar/RBE577_ws/Final/M4Depth/datasets/MidAir" $2