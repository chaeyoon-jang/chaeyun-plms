#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
for data in sst2 qqp qnli mnli
do
    python -m src.main --data-name glue --task-name $data --NO-DEBUG
done