import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from .classification import MultiTaskClassifier

def generate_mask(new_data, dropout=0.0, is_training=False):
    if not is_training:
        dropout_p = 0.0
    new_data = (1 - dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1) - 1)
        new_data[i][one] = 1
    mask = 1.0 / (1 - dropout_p) * torch.bernoulli(new_data)
    mask.requires_grad = False
    return mask

class Dropout(nn.Module):
    def __init__(self, dropout=0):
        super(Dropout, self).__init__()
        self.dropout = dropout
        
    def forward(self, x):
        if self.training == False or self.dropout == 0:
            return x
        if len(x.size()) == 3:
            mask = (1.0 / (1 - self.dropout_p) * torch.bernoulli(
                (1 - self.dropout_p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1)))
            mask.requires_grad = False
            return mask.unsqueeze(1).expand_as(x) * x
        else:
            return F.dropout(x, p=self.dropout, training=self.training)

class SelfAttention(nn.Module):
    def __init__(self, input_size, dropout=None):
        super(SelfAttention, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.dropout = dropout
    
    def forward(self, x, x_mask):
        
        x = self.dropout(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float("inf"))
        alpha = F.softmax(scores, 1)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)

class Pooler(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = Dropout(dropout)
    
    def forward(self, hidden_states):
        logits = hidden_states[:, 0] # cls token's embedding
        logits = self.dropout(logits)
        logits = self.dense(logits)
        pooled_output = torch.tanh(logits)
        return pooled_output

class Attention(nn.Module):
    def __init__(self, x_size, y_size, dropout=None):
        super(Attention, self).__init__()
        assert x_size == y_size
        self.linear = nn.Linear(x_size * 4, 1)
        self.dropout = dropout
    
    def forward(self, x, y, x_mask):
        
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)
        
        flat_x = (
            torch.cat([x, y, x*y, torch.abs(x - y)], 2)\
                .contiguous().view(x.size(0) * x.size(1), -1)
        )
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float("inf"))
        
        return scores

class Classifier(nn.Module):
    def __init__(self, x_size, y_size, dropout=None):
        super(Classifier, self).__init__()
        self.dropout = dropout
        self.proj = nn.Linear(x_size * 4, y_size)
    
    def forward(self, x1, x2, maks=None):
        x = torch.cat([x1, x2], 1)
        x = self.dropout(x)
        scores = self.proj(x)
        return scores