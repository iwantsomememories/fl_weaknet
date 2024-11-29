#!/bin/bash

python AsyncServer.py  --global_round 200 --world_size 11 --num_clients 10 &

for i in $(seq 1 10)
do
    python AsyncClient.py --rank $i --epoch 3 --world_size 11 &
done

wait