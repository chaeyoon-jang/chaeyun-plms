import torch
from torch import nn
from .utils import convert_dict, chaeyun_average
from typing import Optional, Tuple
from transformers import AutoModel

class SingleTaskClassificationModel(nn.Module):
    def __init__(self, config):
        super(SingleTaskClassificationModel, self).__init__()
        self.model = AutoModel.from_pretrained(config.model.name, add_pooling_layer=False)
        self.dense = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        self.dropout = nn.Dropout(config.model.dropout)
        self.out = nn.Linear(config.model.hidden_size, config.dataset.num_classes)
    
    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                ):
        
        outputs = self.model(input_ids, 
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

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

class MultiTaskClassificationModel(nn.Module):
    def __init__(self, config):
        super(MultiTaskClassificationModel, self).__init__()
        self.model = AutoModel.from_pretrained(config.model.name, add_pooling_layer=False)
        self.task_specific_layers = nn.ModuleList()

        for task_def in config.dataset.task.items():
            num_classes = task_def.num_classes
            task_type = task_def.type



    
    def forward(
            self, input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            ):
        return None  



class SingleTaskGenerationModel(nn.Module):
    def __init__(self, config):
        super(SingleTaskClassificationModel, self).__init__()
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
    
model_type = {
    'classification_single': SingleTaskClassificationModel,
    'classification_multi': MultiTaskClassificationModel,
    'generation_single' : SingleTaskGenerationModel
}