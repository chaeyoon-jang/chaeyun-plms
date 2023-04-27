import torch
import torch.nn as nn

class Pooler(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states):
        logits = hidden_states[:, 0, :]
        logits = self.dropout(logits)
        logits = self.dense(logits)
        pooled_output = torch.tanh(logits)
        return pooled_output

class SANClassifier(nn.Module):
    def __init__(self, )


class MultiTaskClassifier:
    def __init__(self, task_def):
        self._task_def = task_def
    
    @classmethod
    def build(hidden_size, task_def, opt):
        proj = nn.Linear(hidden_size, task_def.num_classes)
        task_layer = nn.Sequential(Pooler(hidden_size, task_def.dropout), proj)
        return task_layer