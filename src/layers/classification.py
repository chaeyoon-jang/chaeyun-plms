import torch
import random
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.parameter import Parameter

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
        # if x_mask == 1, -float('inf')
        scores.data.masked_fill_(x_mask==1, -float("inf")) 
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
        scores.data.masked_fill_(x_mask==1, -float("inf"))
        
        return scores

class Classifier(nn.Module):
    def __init__(self, x_size, y_size, dropout=None):
        super(Classifier, self).__init__()
        self.dropout = dropout
        self.proj = nn.Linear(x_size * 4, y_size)
    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        x = self.dropout(x)
        scores = self.proj(x)
        return scores

class SANClassifier(nn.Module):
    """
    Implementation of Stochastic Answer Networks (https://arxiv.org/abs/1804.07888)
    """
    def __init__(self, hidden_size, dropout, num_classes): 
        super(SANClassifier, self).__init__()
        self.dropout = Dropout(dropout)
        self.self_attn = SelfAttention(hidden_size, self.dropout)
        self.attn = Attention(hidden_size, hidden_size, self.dropout)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.K = 5
        self.num_classes = num_classes
        self.classifier = Classifier(hidden_size, self.num_classes, self.dropout)
        self.alpha = Parameter(torch.zeros(1, 1), requires_grad=False)

    def forward(self, hyp_mem, premise_mem, p_mask=None, h_mask=None):
        """_summary_

        Args:
            x (torch.Tensor): The contextual embeddings of the words in premise.
            m_h (torch.Tensor): The contextual embeddings of the words in hypothesis.
            p_mask (torch.Tensor, optional): _description_. Defaults to None.
            h_mask (torch.Tenser, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # self.proj(hyp_mem, premise_mem, hyp_mask, premise_mask)
        s0 = self.self_attn(hyp_mem, h_mask)
        scores_list = []
        for k in range(self.K):
            betas = self.attn(premise_mem, s0, p_mask)
            x0 = torch.bmm(F.softmax(betas, 1).unsqueeze(1), premise_mem).squeeze(1)
            scores = self.classifier(x0, s0)
            scores_list.append(scores)
            
            s0 = self.dropout(s0)
            s0 = self.rnn(x0, s0)            
        mask = generate_mask(
                self.alpha.data.new(x0.size(0), self.num_turn),
                self.mem_random_drop,
                self.training,
            )
        mask = [m.contiguous() for m in torch.unbind(mask, 1)]
        tmp_scores_list = [
                mask[idx].view(x0.size(0), 1).expand_as(inp) * F.softmax(inp, 1)
                for idx, inp in enumerate(scores_list)
            ]
        scores = torch.stack(tmp_scores_list, 2)
        scores = torch.mean(scores, 2)
        scores = torch.log(scores)
        return scores

class MultiTaskClassifier(nn.Module):
    def __init__(self, config):
        super(MultiTaskClassifier, self).__init__()
        self.pooler_layer = Pooler(config.hidden_size, config.dropout)
        if config.enable_san:
            self.proj = SANClassifier(config.hidden_size, config.dropout, config.num_classes)
        else:
            self.proj = nn.Linear(config.hidden_size, config.num_classes) # dropout
        #self.task_layer = nn.Sequential(Pooler(config.hidden_size, config.dropout), self.proj)
        self.max_query = int(config.max_seq_length / 2)
        assert self.max_query > 0
        
    def forward(self, seq_outputs, attention_mask, enable_san):
        if enable_san:
            assert premise_mask is not None
            assert hyp_mask is not None
            hyp_mem      = seq_outputs[:, self.max_query:, :]
            premise_mem  = seq_outputs[:, :self.max_query, :]
            hyp_mask     = attention_mask[:, self.max_query:, :]
            premise_mask = attention_mask[:, :self.max_query, :]
            # forward() takes 2 positional arguments but 5 were given...
            # logits = self.task_layer(seq_outputs, hyp_mem, premise_mask, hyp_mask)
            # logits = self.pooler_layer(seq_outputs)
            logits = self.proj(hyp_mem, premise_mem, hyp_mask, premise_mask)
        else:
            logits = self.pooler_layer(seq_outputs)
            logits = self.proj(logits)
        return logits