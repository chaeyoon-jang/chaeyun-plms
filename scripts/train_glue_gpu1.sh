#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
for data in rte cola mrpc stsb
do
    python -m src.main --data-name glue --task-name $data --NO-DEBUG
done