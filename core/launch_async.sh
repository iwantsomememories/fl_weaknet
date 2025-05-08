#!/bin/zsh

set -e

num_clients=2
global_round=3
world_size=$((num_clients+1))
eval_gap=1
epochs=1
seed=0
float dir_alpha=0.5

model="mnist"
dataset="mnist"
partition="noniid-labeldir"

python3 utils/Datasets.py --dataset "$dataset" --num_clients "$num_clients" --partition "$partition" --dir_alpha "$dir_alpha" --seed "$seed"

wait

python3 AsyncServer.py  --global_round "$global_round" --world_size "$world_size" --num_clients "$num_clients" --eval_gap "$eval_gap" --partition "$partition" --dir_alpha "$dir_alpha" --seed "$seed" --model "$model" &

for i in $(seq 1 "$num_clients")
do
    python3 AsyncClient.py --rank $i --epochs "$epochs" --world_size "$world_size" --num_clients "$num_clients" --dir_alpha "$dir_alpha" --seed "$seed" --model "$model" &
done

wait