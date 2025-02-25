#!/bin/bash

python SemiAsyncServer.py  --global_round 3 --world_size 3 --num_clients 2 --eval_gap 2 &

for i in $(seq 1 2)
do
    python SemiAsyncClient.py --rank $i --epochs 2 --world_size 3 &
done

wait