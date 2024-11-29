#!/bin/bash

python AsyncServer.py  --global_round 20 --world_size 6 --num_clients 5 &

for i in $(seq 1 5)
do
    python AsyncClient.py --rank $i --epoch 2 --world_size 6 &
done

wait