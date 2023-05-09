import torch
import torch.nn as nn
import torch.nn.functional as F 
from layers import Dropout, SelfAttention, Attention,\
    Classifier, Pooler, generate_mask
from torch.nn.parameter import Parameter

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

    def forward(self,  x0, h0, p_mask=None, h_mask=None):
        """_summary_

        Args:
            x (torch.Tensor): The contextual embeddings of the words in premise.
            m_h (torch.Tensor): The contextual embeddings of the words in hypothesis.
            p_mask (torch.Tensor, optional): _description_. Defaults to None.
            h_mask (torch.Tenser, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        s0 = self.self_attn(x0, h_mask)
        scores_list = []
        for k in range(self.K):
            betas = self.attn(x0, s0, p_mask)
            x0 = torch.bmm(F.softmax(betas, 1).unsqueeze(1), x0).squeeze(1)
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
        if config.enable_san:
            self.proj = SANClassifier(config.hidden_size, config.dropout, config.num_classes)
        else:
            self.proj = nn.Linear(config.hidden_size, config.dropout, config.num_classes)
        self.task_layer = nn.Sequential(Pooler(config.hidden_size, config.dropout), self.proj)
    
    def forward(self, seq_outputs, premise_mask, hyp_mask, enable_san):
        if enable_san:
            max_query = hyp_mask.size(1)
            assert max_query > 0
            assert premise_mask is not None
            assert hyp_mask is not None
            hyp_mem = seq_outputs[:, :max_query, :]
            logits = self.task_layer(seq_outputs, hyp_mem, premise_mask, hyp_mask)
        else:
            logits = self.task_layer(seq_outputs)
        
        return logits