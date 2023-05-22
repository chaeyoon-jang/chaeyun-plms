import os
import os.path as p

import torch
import datasets
from .utils import seed_worker
from torch.utils.data import Dataset, DataLoader,\
    ConcatDataset, random_split, BatchSampler

task_text_field_map = {
     'cola': ['sentence'],
     'sst2': ['sentence'],
     'mrpc': ['sentence1', 'sentence2'],
     'qqp': ['question1', 'question2'],
     'stsb': ['sentence1', 'sentence2'],
     'mnli': ['premise', 'hypothesis'],
     'qnli': ['question', 'sentence'],
     'rte': ['sentence1', 'sentence2'],
     'wnli': ['sentence1', 'sentence2']}

glue_task_num_labels = {
     'cola': 2, 'sst2': 2,
     'mrpc': 2, 'qqp': 2,
     'stsb': 1, 'mnli': 3,
     'qnli': 2, 'rte': 2,
     'wnli': 2}

loader_columns = [
     'input_ids',
     'attention_mask',
     'label',
     'premise_mask',
     'hyp_mask',
     'task_name']

large_task = ['mnli', 'qqp', 'sst2', 'qnli']
    
def make_glue_dataloader_v1(dataframe, tokenizer, config):
    
    padding = 'max_length' if config.pad_to_max_length else False

    text_fields = task_text_field_map[config.name]
    num_labels = glue_task_num_labels[config.name]
    
    generator = torch.Generator()
    generator.manual_seed(0)

    def convert_to_features(example_batch, indices=None):
        
        if len(text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[text_fields[0]], 
                                               example_batch[text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[text_fields[0]]
                
        features = tokenizer.batch_encode_plus(texts_or_text_pairs,
                                               max_length=config.max_seq_length,
                                               padding=padding,
                                               add_special_tokens=True,
                                               return_token_type_ids=True) 
            
        features['label'] = example_batch['label']
        return features
       
    for split in dataframe.keys():
        dataframe[split] = dataframe[split].map(convert_to_features, batched=True)
        columns = [c for c in dataframe[split].column_names if c in loader_columns]
        dataframe[split].set_format(type="torch", columns=columns)

    eval_splits = [x for x in dataframe.keys() if 'validation' in x]

    if config.name in large_task:
         data_size = len(dataframe['train'])
         validation_size = 1000
         train_data, validation_data = random_split(dataframe['train'], [data_size - validation_size, validation_size], generator=generator)
    
    else:
         train_data = dataframe['train']

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    
    train_loader = DataLoader(train_data,
                              batch_size = config.batch_size,
                              num_workers = config.num_workers,
                              sampler=train_sampler,
                              worker_init_fn=seed_worker,
                              shuffle=False)
    
    if len(eval_splits) == 1: 
            if config.name not in large_task:
                data_size = len(dataframe['validation'])
                validation_size = int(data_size * 0.5)
                test_size = data_size - validation_size
                validation_data, test_data = random_split(dataframe['validation'], [validation_size, test_size], generator=generator)
            
            else:
                 test_data = dataframe['validation']

            eval_loader = DataLoader(validation_data,
                                     batch_size = config.batch_size,
                                     num_workers = config.num_workers)
            
            test_loader = DataLoader(test_data,
                                     batch_size = config.batch_size,
                                     num_workers = config.num_workers)
            
    elif len(eval_splits) > 1:
           eval_loader = DataLoader(validation_data,
                                    batch_size = config.batch_size,
                                    num_workers = config.num_workers)
           
           test_loader = [DataLoader(dataframe[x],
                                     batch_size= config.batch_size,
                                     num_workers= config.num_workers) for x in eval_splits]
    
    torch.save(train_loader, os.path.join(config.save_dir, f'{config.name}_train_loader_v1.pth'))
    torch.save(eval_loader,  os.path.join(config.save_dir, f'{config.name}_valid_loader_v1.pth'))
    torch.save(test_loader,  os.path.join(config.save_dir, f'{config.name}_test_loader_v1.pth'))

def make_glue_dataloader_v2(dataframe, tokenizer, config):
    
    padding = 'max_length' if config.pad_to_max_length else False

    text_fields = task_text_field_map[config.name]
    num_labels = glue_task_num_labels[config.name]

    def convert_to_features(example_batch, indices=None):
        
        if len(text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[text_fields[0]], 
                                               example_batch[text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[text_fields[0]]
                
        features = tokenizer.batch_encode_plus(texts_or_text_pairs,
                                               max_length=config.max_seq_length,
                                               padding=padding,
                                               add_special_tokens=True,
                                               return_token_type_ids=True) 
            
        features['label'] = example_batch['label']
        return features
       
    for split in dataframe.keys():
        dataframe[split] = dataframe[split].map(convert_to_features, batched=True)
        columns = [c for c in dataframe[split].column_names if c in loader_columns]
        dataframe[split].set_format(type="torch", columns=columns)

    eval_splits = [x for x in dataframe.keys() if 'validation' in x]
    test_splits = [x for x in dataframe.keys() if 'test' in x]

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataframe['train'])
    
    train_loader = DataLoader(dataframe['train'],
                              batch_size = config.batch_size,
                              num_workers = config.num_workers,
                              sampler=train_sampler,
                              worker_init_fn=seed_worker,
                              shuffle=False)
    
    if len(eval_splits) == 1: 

            eval_loader = DataLoader(dataframe['validation'],
                                     batch_size = config.batch_size,
                                     num_workers = config.num_workers)
            
            test_loader = DataLoader(dataframe['test'],
                                     batch_size = config.batch_size,
                                     num_workers = config.num_workers)
            
    elif len(eval_splits) > 1:
           eval_loader = [DataLoader(dataframe[x],
                                     batch_size= config.batch_size,
                                     num_workers= config.num_workers) for x in eval_splits]
           
           test_loader = [DataLoader(dataframe[x],
                                     batch_size= config.batch_size,
                                     num_workers= config.num_workers) for x in test_splits]
    
    torch.save(train_loader, os.path.join(config.save_dir, f'{config.name}_train_loader_v2.pth'))
    torch.save(eval_loader,  os.path.join(config.save_dir, f'{config.name}_valid_loader_v2.pth'))
    torch.save(test_loader,  os.path.join(config.save_dir, f'{config.name}_test_loader_v2.pth'))

def make_multi_glue_dataloader(tokenizer, config):
    
    # Case 1)
    # batch size / 2 -> preprocessing
    # dist = 2 
    # as a result total update 1 batch is 2 * batch_size / 2 
    
    # Case 2)
    # make multiple dataloaders for epch task.
    
    padding = 'max_length' if config.pad_to_max_length else False
    task_names = task_text_field_map.keys()
    dataframes = dict()
    
    for name in task_names:
        dataframes[name] = datasets.load_dataset('glue', name)
    
    def preprocessing(dataframe, task):
        
        text_fields = task_text_field_map[task]
        num_labels = glue_task_num_labels[task]
        
        premise_mask = [0 for _ in range(256)] + [1 for _ in range(256)]
        hyp_mask = [1 for _ in range(256)] + [0 for _ in range(256)]
        
        def convert_to_features(example_batch, indices=None):

            if len(text_fields) > 1:
                temp_seq_length = int(config.max_seq_length/2)
                premise = tokenizer.batch_encode_plus(example_batch[text_fields[0]],
                                                      max_length=temp_seq_length,
                                                      padding=padding,
                                                      add_special_tokens=True,
                                                      truncation=True)
                hypothesis = tokenizer.batch_encode_plus(example_batch[text_fields[1]],
                                                      max_length=temp_seq_length,
                                                      padding=padding,
                                                      add_special_tokens=True,
                                                      truncation=True)
                
                total_length = len(premise['input_ids'])
                features = {'input_ids': [premise['input_ids'][i] + hypothesis['input_ids'][i] for i in range(total_length)],
                            'attention_mask': [premise['attention_mask'][i] + hypothesis['attention_mask'][i] for i in range(total_length)]}
                
            else:
                texts_or_text_pairs = example_batch[text_fields[0]]
                features = tokenizer.batch_encode_plus(texts_or_text_pairs,
                                                    max_length=config.max_seq_length,
                                                    padding=padding,
                                                    add_special_tokens=True,
                                                    return_token_type_ids=True,
                                                    truncation=True)

            features['label'] = example_batch['label'] #TODO: stsb label type check
            features['task_name'] = [task for _ in range(len(example_batch['label']))]
            features['premise_mask'] = [premise_mask for _ in range(len(example_batch['label']))]
            features['hyp_mask'] = [hyp_mask for _ in range(len(example_batch['label']))]
            return features
        
        for split in dataframe.keys():
            dataframe[split] = dataframe[split].map(convert_to_features, batched=True)
            columns = [c for c in dataframe[split].column_names if c in loader_columns]
            dataframe[split].set_format(type="torch", columns=columns)

        eval_splits = [x for x in dataframe.keys() if 'validation' in x]
        test_splits = [x for x in dataframe.keys() if 'test' in x]
        
        if task != 'mnli':
            eval_splits = None
            test_splits = None
            
        return dataframe, eval_splits, test_splits
    
    train_data  = dict()
    eval_loader = dict()
    test_loader = dict()
    
    train_length = 0
    valid_length = 0
    test_length  = 0
    
    for df_name in dataframes.keys():
        df, e_s, t_s = preprocessing(dataframes[df_name], df_name) 
        
        if e_s is not None:
            validset = e_s
            testset = t_s
        
        '''
        train_data[df_name] = list(BatchSampler(df['train'],
                                               batch_size = int(config.batch_size / 2),
                                               drop_last=True))
        '''
        train_data = df['train'] #default: Concat datasets
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
        train_loader = DataLoader(train_data,
                                  batch_size = config.batch_size, 
                                  num_workers = config.num_workers,
                                  sampler = train_sampler,
                                  worker_init_fn = seed_worker) 
        train_data[df_name] = train_loader
        
        try:
            train_length += len(df['train']) 
            valid_length += len(df['validation']) 
            test_length  += len(df['test']) 
            
            eval_loader[df_name]  = DataLoader(df['validation'],
                                              batch_size = config.batch_size,
                                              num_workers = config.num_workers)
            test_loader[df_name]  = DataLoader(df['test'],
                                               batch_size = config.batch_size,
                                               num_workers = config.num_workers)
            
            eval_loader[df_name] = df['validation']
            test_loader[df_name] = df['test']
            
        except:
            train_length += len(df['train']) 
            valid_length += len(validset[0]) 
            test_length  += len(testset[0]) 
            
            eval_loader[df_name] = DataLoader(validset[0],
                                             batch_size = config.batch_size,
                                             num_workers = config.num_workers) #TODO: mismatch or match?
            test_loader[df_name]  = DataLoader(testset[0],
                                            batch_size = config.batch_size,
                                             num_workers = config.num_workers)
            
            eval_loader[df_name+'_match'] = validset[0]
            eval_loader[df_name+'_mismatch'] = validset[1]

            test_loader[df_name+'_match'] = testset[0]
            eval_loader[df_name+'_mismatch'] = testset[1]
    

    print(f'The length of total train data is {train_length}')
    print(f'The length of total valid data is {valid_length}')
    print(f'The length of total test data is {test_length}')
    
    torch.save(train_data, os.path.join(config.save_dir, f'multi-task_train_loader.pth'))
    torch.save(eval_loader,  os.path.join(config.save_dir, f'multi-task_valid_loader.pth'))
    torch.save(test_loader,  os.path.join(config.save_dir, f'multi-task_test_loader.pth'))   

# TODO: preprocessing function for squad dataset.
# def make_squad_dataloader(dataframe, tokenizer, config): 