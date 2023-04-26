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
* data  : `0.5 * dev_set`
* fine-tuning method : `single-task finetuning & full-fine tuning`

| Method             | RTE    | MRPC   | COLA   | SST-2  | STS-B | MNLI | QQP  | QNLI
| :-                 | :-:    | :-:    | :-:    | :-:    | :-:   | :-:  | :-:  | :-:
| AdamW              | 75.12  | 91.99  | 64.95  | 94.86  | 61.84 |  -   |  -   |  -
| + SWA              | 81.37  | 92.54  | 60.85  | 94.97  | 77.87 |  -   |  -   |  -