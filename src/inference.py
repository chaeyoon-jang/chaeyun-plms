import os
import os.path as p

import torch
import argparse
import datasets

from transformers import AutoTokenizer
from .train import validate
from .utils import chaeyun_load, detach_module
from .utils.metric import glue_metrics
from .model import build_model
from .utils import configure_cudnn
from .utils.dataset import load_glue_data_loader

import torch
from torch.utils.data import Dataset, DataLoader

def get_arg_parser():

    parser = argparse.ArgumentParser(
        description="Code for evaluating any PLMs provided by HuggingFace on Benchmark tasks")
    
    parser.add_argument('--ckpt-path', type=str, default='', action='')
    parser.add_argument('--data-name', '-d', type=str, default='glue')
    parser.add_argument('--task-name', '-t', type=str, default='cola')
    parser.add_argument('--is-swa', '-s', type=bool, default=False)

    return parser

def evaluate():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = get_arg_parser()
    args = parser.parse_args()
    default_path = ['./finetuning', f'{args.data_name}-base', f'{args.task_name}.yaml']
    config_path = os.path.join(*default_path)

    swa_path = None
    if args.is_swa:
        swa_path = ['./finetuning', f'{args.data_name}-swa', f'{args.task_name}.yaml']
        swa_path = os.path.join(*swa_path)

    config = chaeyun_load(args.config_path, swa_path)
    configure_cudnn(True)

    print("Loading Model...")
    model = build_model(config)
    state_dict = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(detach_module(state_dict))
    model.to(device)
    
    print("Loading Test dataset...")
        
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    if args.data_name == 'glue':
        flag = ''
        if 'mnli' in args.task_name:
            flag = args.task_name.replace('mnli', '')
            args.task_name = 'mnli'
        
        test_data_path = p.join(config.dataset.save_dir, f'{args.task_name}_test{flag}.pth')
        
        if not p.isfile(test_data_path):
            os.makedirs(config.dataset.save_dir, exist_ok=True)
            make_glue_dataframe(args.task_name, tokenizer, config.dataset)
        
        test_ds = GLUEDataset(torch.load(test_data_path))
        test_loader = load_glue_data_loader(test_ds, batch_size=config.dataset.batch_size)
        
        metric = glue_metrics(config.dataset.name)        
    
    print("Start testing...")
    _, acc, metric = validate(model, None,
                           metric, test_loader, 
                           config.dataset.name, device)

if __name__ == '__main__':
    evaluate()