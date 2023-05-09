import datasets
import torch
from torch import nn

class chaeyun_criterion(nn.Module):
    def __init__(self, type):
        super(chaeyun_criterion, self).__init__()
        self.loss_fn =  torch.nn.MSELoss() if type == 'regression' else torch.nn.CrossEntropyLoss()
        self.type = type
    
    def forward(self, logits, targets):

        if type == 'generation':
            logits  = logits[..., :-1, :].contiguous()
            targets = targets[..., 1:].contiguous()
        
        if type == 'regression':
            logits = logits.to(torch.float32)
            targets = targets / 5.0
            targets = targets.to(torch.float32)

        return self.loss_fn(logits, targets)

class glue_metrics:
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

class multi_glue_metrics:
    def __init__(self):
        self.type1 = glue_metrics('cola')
        self.type2 = glue_metrics('stsb')
        self.type3 = glue_metrics('mrpc')
        self.type4 = glue_metrics('sst2')
    
    def calculate(self, logits, targets, metric_type):
        if metric_type == 1:
            return self.type1(logits, targets)
        elif metric_type == 2:
            return self.type2(logits, targets)
        elif metric_type == 3:
            return self.type3(logits, targets)
        else:
            return self.type4(logits, targets)