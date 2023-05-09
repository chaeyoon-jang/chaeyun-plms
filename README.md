# Pretrained Language Models (PLMs) Baseline
This repository contains the code for fine-tuning any PLMs provided by HuggingFace on Benchmark tasks.

<b>Note:</b> the baselines provided in this repository aim to contribute to the research on generalization by exploring the loss surfaces of PLMs. Accordingly, [stochastic weight averaging (SWA)](https://arxiv.org/abs/1803.05407) and [sharpness aware minimization (SAM)](https://arxiv.org/abs/2010.01412) are also provided as optimization methods.

# Dependencies
* [PyTorch](http://pytorch.org/)
* [Transformers](https://huggingface.co/docs/transformers/)
* [Datasets](https://huggingface.co/docs/datasets/)

# Usage
You can train the endpoints using the following command.

```bash
sh ./scripts/train_glue_gpu1.sh --data-type=<BENCHMARK> \
                                --task-type=<TASK_TYPE> \
                                --trainig-type=<TRAIN_TYPE> \
                                --is-swa=<SWA_OR_NOT> \
                                --NO-DEBUG \
```

Parameters:

* ```BENCHMARK``` &mdash; benchmark name [glue/squad] (default: glue)
* ```DATASET``` &mdash; task name for glue datasets [cola/rte/mnli/qnli/qqp/mrpc/stsb/sst2]
* ```TRAIN_TYPE``` &mdash; fine-tuning type [single-task/multi-task]
* ```SWA_OR_NOT``` &mdash; turn on swa mode or not

Use the `--DEBUG` flag if you want to debug your codes.

# Example Results

**[GLUE (Wang et al., 2019)](https://gluebenchmark.com/)**

* model : `roberta-base`
* data  : `dev_set`
* fine-tuning method : `single-task finetuning & full-fine tuning`

| Method             | RTE    | MRPC   | COLA   | SST-2  | STS-B | QNLI | QQP   | MNLI(M) | MNLI(MM)
| :-                 | :-:    | :-:    | :-:    | :-:    | :-:   | :-:  | :-:   | :-:     | :-:
| AdamW              | 82.22  | 93.32  | 62.27  | 94.77  | 76.52 |  -   | 88.25 |  -      | -
| + SWA (1 Budget)   | 80.90  | 93.15  | 62.04  | 95.45  | 76.54 |  -   | 88.13 |  -      | -
| + SWA (3 Budgets)  | 83.33  | 93.15  | 62.04  | 95.45  | 76.54 |  -   | 88.18 |  -      | -
| Ensemble (n=3)     | 83.33  | 93.17  | 63.04  | -      | -     |  -   |  -    |  -      | -