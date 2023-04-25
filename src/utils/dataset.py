import torch
from .utils import seed_worker
from torch.utils.data import Dataset, DataLoader, random_split

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

large_task = [
     'mnli',
     'qqp',
     'sst2',
     'qnli'
]

def make_dataloader(batch_size,
                    dataframe, num_workers,
                    task,
                    tokenizer, 
                    padding,
                    max_len,
                    seed,
                    ):
       
    text_fields = task_text_field_map[task]
    num_labels = glue_task_num_labels[task]
    
    generator = torch.Generator()
    generator.manual_seed(0)

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

    if task in large_task:
         data_size = len(dataframe['train'])
         validation_size = 1000
         train_data, validation_data = random_split(dataframe['train'], [data_size - validation_size, validation_size], generator=generator)
    
    else:
         train_data = dataframe['train']

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    
    train_loader = DataLoader(
            train_data,
            batch_size = batch_size,
            num_workers = num_workers,
            sampler=train_sampler,
            worker_init_fn=seed_worker,
            shuffle=False,
            )
    
    if len(eval_splits) == 1:
            
            if task not in large_task:
                data_size = len(dataframe['validation'])
                validation_size = int(data_size * 0.5)
                test_size = data_size - validation_size
                validation_data, test_data = random_split(dataframe['validation'], [validation_size, test_size], generator=generator)
            
            else:
                 test_data = dataframe['validation']

            eval_loader = DataLoader(
                validation_data,
                batch_size = batch_size,
                num_workers = num_workers,
                )
            
            test_loader = DataLoader(
                test_data,
                batch_size = batch_size,
                num_workers = num_workers,
                )
            
    elif len(eval_splits) > 1:

           eval_loader = DataLoader(
                validation_data,
                batch_size = batch_size,
                num_workers = num_workers,
                )
           
           test_loader = [
                DataLoader(
                dataframe[x],
                batch_size=batch_size,
                num_workers=num_workers,
                ) for x in eval_splits]
    
    return train_loader, eval_loader, test_loader