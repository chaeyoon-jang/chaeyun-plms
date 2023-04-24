import datasets
import torch

class metrics:
    def __init__(self, task_flag):
        self.metric_fn = datasets.load_metric('glue', task_flag)
        self.task_flag = task_flag

    def calculate(self, logits, targets):

        if self.task_flag != 'stsb':
            _, preds = torch.max(logits, dim=-1)
            result = preds.eq(targets).sum().item()
            result = (result / preds.size(0)) * 100
            metric = self.metric_fn.compute(predictions=preds, references=targets)
        else:
            metric = self.metric_fn.compute(predictions=logits, references=targets)
        
        if self.task_flag == 'cola':
            metric = metric["matthews_correlation"]
        elif self.task_flag == 'stsb':
            metric = metric["pearson"]
            result = metric
        elif self.task_flag in ['mrpc', 'qqp']:
            metric = metric["f1"]
        else:
            metric = result

        return result, metric