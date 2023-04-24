import torch
from torch import nn
import transformers
from typing import Optional
from transformers import RobertaModel

class RobertaGLUE(nn.Module):
    def __init__(self, config):
        super(RobertaGLUE, self).__init__()
        self.roberta = RobertaModel.from_pretrained(config['model_type'], add_pooling_layer=False)
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['classifier_dropout'])
        self.out = nn.Linear(config['hidden_size'], config['num_labels'])
    
    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                ):
        
        outputs = self.roberta(input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        logits = sequence_output[:, 0, :]  
        logits = self.dropout(logits)
        logits = self.dense(logits)
        logits = torch.tanh(logits)
        logits = self.dropout(logits)
        logits = self.out(logits)

        return {
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attention' : outputs.attentions,
        }