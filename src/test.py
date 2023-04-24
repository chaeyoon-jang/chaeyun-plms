import torch
import datasets
from transformers import RobertaTokenizer
from .train import validate
from .utils.metric import metrics
from .model import RobertaGLUE
from collections import OrderedDict
from .utils import set_seed, configure_cudnn

import torch
from torch.utils.data import Dataset, DataLoader

task_text_field_map = {
            'cola': ['sentence'],
            'sst2': ['sentence'],
            'mrpc': ['sentence1', 'sentence2'],
            'qqp': ['question1', 'question2'],
            'stsb': ['sentence1', 'sentence2'],
            'mnli': ['premise', 'hypothesis'],
            'qnli': ['question', 'sentence'],
            'rte': ['sentence1', 'sentence2'],
            'wnli': ['sentence1', 'sentence2'],
            'ax': ['premise', 'hypothesis']
        }

glue_task_num_labels = {
            'cola': 2,
            'sst2': 2,
            'mrpc': 2,
            'qqp': 2,
            'stsb': 1,
            'mnli': 3,
            'qnli': 2,
            'rte': 2,
            'wnli': 2,
            'ax': 3
        }

loader_columns = [
            'input_ids',
            'token_type_ids',
            'attention_mask',
            'label'
        ]

def make_dataloader(batch_size,
                    dataframe, num_workers,
                    task,
                    tokenizer, 
                    padding,
                    max_len):
       
    text_fields = task_text_field_map[task]
    num_labels = glue_task_num_labels[task]
       
    def convert_to_features(example_batch, indices=None):
        
        if len(text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[text_fields[0]], 
                                                example_batch[text_fields[1]]))

        else:
            texts_or_text_pairs = example_batch[text_fields[0]]
                
        features = tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=max_len,
            padding=padding,
            add_special_tokens=True,
            return_token_type_ids=True
        ) 
            
        features['label'] = example_batch['label']
        return features
       
    for split in dataframe.keys():
        dataframe[split] = dataframe[split].map(
            convert_to_features,
            batched=True,
            )
        columns = [c for c in dataframe[split].column_names if c in loader_columns]
        dataframe[split].set_format(type="torch", columns=columns)

    eval_splits = [x for x in dataframe.keys() if 'validation' in x]
    
    train_loader = DataLoader(
            dataframe['train'],
            batch_size = batch_size,
            num_workers = num_workers
            )
    if len(eval_splits) == 1:
            eval_loader = DataLoader(
                dataframe['validation'],
                batch_size = batch_size,
                num_workers = num_workers,
                )
            test_loader = DataLoader(
                dataframe['test'],
                batch_size = batch_size,
                num_workers = num_workers,
                )
            
    elif len(eval_splits) > 1:
           eval_loader = [
                DataLoader(
                dataframe[x],
                batch_size=batch_size,
                num_workers=num_workers,
                ) for x in eval_splits]
           
           test_loader = [
                DataLoader(
                dataframe[x],
                batch_size=batch_size,
                num_workers=num_workers,
                ) for x in eval_splits]
    
    return train_loader, eval_loader, test_loader

def evaluate():

    set_seed(42)
    configure_cudnn(True)

    print("Loading Model...")
    config = {}
    config['model_type'] = 'roberta-base'
    config['classifier_dropout'] = 0.1
    config['num_labels'] = 2
    config['hidden_size'] = 768

    model = RobertaGLUE(config)

    state_dict = torch.load('/mnt/home/chaeyun-jang/roberta/complete_for_vision/cola/cola_2e-05_42.pt', map_location='cpu')

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict['model_state_dict'].items():
         if k == 'n_averaged':
              continue
         if 'module' in k:
              name = k[14:]
              new_state_dict[name] = v
         else:
              name = k
              new_state_dict[name] = v
    
    #model.load_state_dict(state_dict['model_state_dict'])

    model.load_state_dict(new_state_dict)
    print("Loading Test dataset...")
    data = datasets.load_dataset('glue', 'cola')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_len = 200
    padding = 'max_length'
    _, valid_loader, _ = make_dataloader(
        32,
        data, 1,
        'cola', tokenizer, 
        padding, max_len)
    metric = metrics('cola')
    _, acc, add = validate(model,
             None,
             metric,
             valid_loader, 
             'cola',
             'cpu')
    print(acc)
    print(add)

if __name__ == '__main__':
    evaluate()