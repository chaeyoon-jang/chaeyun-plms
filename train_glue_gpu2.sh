#!/bin/bash
for data in sst2 qqp qnli mnli
do
    python -m src.main --data-type glue --task-type $data --NO-DEBUG
done