#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
for data in rte cola mrpc stsb
do
    python -m src.main --data-type glue --task-type $data --NO-DEBUG
done