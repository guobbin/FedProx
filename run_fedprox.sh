#!/usr/bin/env bash
# synthetic
#python3  -u main.py --dataset=$1 --optimizer='fedprox'  \
#            --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 \
#            --eval_every=1 --batch_size=10 \
#            --num_epochs=20 \
#            --model='mclr' \
#            --drop_percent=$2 \
#            --mu=$3 \

# mnist
python3  -u main.py --dataset=$1 --optimizer='fedprox'  \
            --learning_rate=0.03 --num_rounds=100 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --model='mclr' \
            --drop_percent=$2 \
            --mu=$3 \

