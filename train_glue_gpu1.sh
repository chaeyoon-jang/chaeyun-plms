#!/bin/bash
for data in rte cola mrpc stsb
do
    python -m src.main --data-type glue --task-type $data --NO-DEBUG
done