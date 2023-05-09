import os
import os.path as p

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
     'wnli': ['sentence1', 'sentence2']}

glue_task_num_labels = {
     'cola': 2, 'sst2': 2,
     'mrpc': 2, 'qqp': 2,
     'stsb': 1, 'mnli': 3,
     'qnli': 2, 'rte': 2,
     'wnli': 2}

loader_columns = [
     'input_ids',
     'token_type_ids',
     'attention_mask',
     'label']

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

    task_names = task_text_field_map.keys()
    dataframes = dict()
    for name in task_names:
        dataframes[name] = datasets.load_dataset('glue', name)
    
    def preprocessing(dataframe, task):
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
                                                return_token_type_ids=True,
                                                truncation=True)
            
            features['label'] = example_batch['label'] #TODO: stsb label type check
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
    
    #final_dataframe = {'train':[], 'test':[], 'validation':[]}
    train_data = list()
    eval_loader = dict()
    test_loader  = dict()
    for df_name in dataframes.keys():
        df, e_s, t_s = preprocessing(dataframes[df_name], df_name) 
        
        if e_s is not None:
            validset = e_s
            testset = t_s
            
        train_data = train_data + df['train']
        
        try:
            eval_loader[df_name] = DataLoader(df['validation'],
                                              batch_size = config.batch_size,
                                              num_workers = config.num_workers)
            test_loader[df_name]  = DataLoader(df['test'],
                                               batch_size = config.batch_size,
                                               num_workers = config.num_workers)
        except:
            valid_data[df_name] = DataLoader(validset[0],
                                             batch_size = config.batch_size,
                                             num_workers = config.num_workers) #TODO: mismatch or match?
            test_data[df_name]  = DataLoader(testset[0],
                                             batch_size = config.batch_size,
                                             num_workers = config.num_workers)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    
    train_loader = DataLoader(train_data,
                              batch_size = config.batch_size,
                              num_workers = config.num_workers,
                              sampler=train_sampler,
                              worker_init_fn=seed_worker,
                              shuffle=True) #TODO: shuffle order check
    
    torch.save(train_loader, os.path.join(config.save_dir, f'{config.name}_train_loader.pth'))
    torch.save(eval_loader,  os.path.join(config.save_dir, f'{config.name}_valid_loader.pth'))
    torch.save(test_loader,  os.path.join(config.save_dir, f'{config.name}_test_loader.pth'))   

# TODO: preprocessing function for squad dataset.
# def make_squad_dataloader(dataframe, tokenizer, config): 