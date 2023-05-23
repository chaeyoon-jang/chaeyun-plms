import torch
from torch import nn
from typing import Optional, Tuple
from transformers import AutoModel

from .utils import convert_dict

class SequenceClassificationModel(nn.Module):
    def __init__(self, config):
        super(SequenceClassificationModel, self).__init__()
        self.model = AutoModel.from_pretrained(config.model.name, add_pooling_layer=False)
        self.dense = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        self.dropout = nn.Dropout(config.model.dropout)
        self.out = nn.Linear(config.model.hidden_size, config.dataset.num_classes)
    
    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None
                ):

        outputs = self.model(input_ids, 
                             attention_mask=attention_mask)

        sequence_output = outputs[0]
        logits = sequence_output[:, 0, :]  
        logits = self.dropout(logits)
        logits = self.dense(logits)
        logits = torch.tanh(logits)
        logits = self.dropout(logits)
        logits = self.out(logits)

        return convert_dict({
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attention': outputs.attentions,
        })

class TextGenerationModel(nn.Module):
    def __init__(self, config):
        super(TextGenerationModel, self).__init__()
        self.lm_model = AutoModel.from_pretrained(config.model.name, add_pooling_layer=False)
        self.lm_head = nn.Linear(config.model.n_embs, config.model.vocab_size, bias=False)
    
    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                ):
        
        outputs = self.lm_model(input_ids, 
                             past_key_values=past_key_values,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        
        sequence_output = outputs[0]
        lm_logits = self.lm_head(sequence_output)

        return convert_dict({
            'logits': lm_logits,
            'past_key_values': outputs.past_key_values,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'cross_attentions' : outputs.cross_attentions,
        })

def build_model(config) -> nn.Module:
    name = config.model.name
    if 'roberta' in name:
        model = SequenceClassificationModel(config)
    
    elif 't5' in name:
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(name)
    
    elif 'gpt' in name:
        model = TextGenerationModel(config)
    
    else:
        raise NotImplementedError(f'Unknown config.model.name = \"{name}\"')
    return model