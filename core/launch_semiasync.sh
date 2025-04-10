#!/bin/bash

declare -i num_clients=2
declare -i global_round=10
declare -i world_size=$((num_clients+1))
declare -i eval_gap=2
declare -i epochs=2
declare -i dir_alpha=0.5
declare -i seed=0

model="mnist"
dataset="mnist"
partition="noniid-labeldir"

python3 utils/Datasets.py --dataset "$dataset" --num_clients "$num_clients" --partition "$partition" --dir_alpha "$dir_alpha" --seed "$seed"

wait

python3 SemiAsyncServer.py  --global_round "$global_round" --world_size "$world_size" --num_clients "$num_clients" --eval_gap "$eval_gap" --partition "$partition" --dir_alpha "$dir_alpha" --seed "$seed" --model "$model" &

for i in $(seq 1 "$num_clients")
do
    python3 SemiAsyncClient.py --rank $i --epochs "$epochs" --world_size "$world_size" --num_clients "$num_clients" --dir_alpha "$dir_alpha" --seed "$seed" --model "$model" &
done

wait