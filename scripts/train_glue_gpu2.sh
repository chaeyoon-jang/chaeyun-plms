#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
for data in sst2 qqp qnli mnli
do
    python -m src.main --data-type glue --task-type $data --NO-DEBUG
done