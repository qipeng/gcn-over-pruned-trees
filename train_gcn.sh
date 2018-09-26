#!/bin/bash

SAVE_ID=$1
python train.py --id $SAVE_ID --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003
