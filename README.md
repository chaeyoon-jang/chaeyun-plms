# Roberta-GLUE Baseline
This repository contains the code for fine-tuning the Roberta-base model for the GLUE task.

# Dependencies
* [PyTorch](http://pytorch.org/)
* [Transformers](https://huggingface.co/docs/transformers/)
* [Datasets](https://huggingface.co/docs/datasets/)

# Usage

# Results

**[GLUE (Wang et al., 2019)](https://gluebenchmark.com/)**

* model : `roberta-base`
* data  : `0.5 * dev_set`
* fine-tuning : `single-task finetuning & full-fine tuning`

| Method             | RTE    | MRPC   | COLA   | SST-2  | STS-B | MNLI | QQP  | QNLI
| :-                 | :-:    | :-:    | :-:    | :-:    | :-:   | :-:  | :-:  | :-:
| AdamW              | 75.12  | 91.99  | 64.95  | 94.86  | 61.84 |  -   |  -   |  -
| + SWA              | 81.37  | 92.54  | 60.85  | 94.97  | 77.87 |  -   |  -   |  -