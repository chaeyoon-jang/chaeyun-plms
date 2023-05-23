import torch
import datasets
import os.path as p

glue_task_cols_map = {
    'ax': ['premise', 'hypothesis'],
     'cola': ['sentence'],
     'sst2': ['sentence'],
     'mrpc': ['sentence1', 'sentence2'],
     'qqp': ['question1', 'question2'],
     'stsb': ['sentence1', 'sentence2'],
     'mnli': ['premise', 'hypothesis'],
     'qnli': ['question', 'sentence'],
     'rte': ['sentence1', 'sentence2'],
     'wnli': ['sentence1', 'sentence2']}

loader_columns = [
    'input_ids',
    'attention_mask',
    'label'    
]

def make_glue_dataframe(task_name, tokenizer, config):
    
    df = datasets.load_dataset('glue', task_name)
    text_fields = glue_task_cols_map[task_name]
    
    def convert_to_features(example):
        padding ='max_length' if config.pad_to_max_length else False
        features = dict()
        if len(text_fields) > 1:
            texts_or_text_pairs = list(zip(example[text_fields[0]],
                                           example[text_fields[1]]))
        else:
            texts_or_text_pairs = example[text_fields[0]]
            
        features = tokenizer.batch_encode_plus(texts_or_text_pairs,
                                               max_length=config.max_seq_length,
                                               padding=padding,
                                               truncation=True,
                                               add_special_tokens=True,
                                               return_token_type_ids=False)
        features['label'] = example['label']
        return features
    
    for split in df.keys():
        df[split] = df[split].map(convert_to_features, batched=True)
        columns = [c for c in df[split].column_names if c in loader_columns]
        df[split].set_format(type='torch', columns=columns)
        torch.save(df[split], p.join(config.save_dir, f'{task_name}_{split}.pth'))