import os
import os.path as p

import torch
import argparse
import datasets

from transformers import AutoTokenizer
from .train import validate
from .utils import chaeyun_load, detach_module
from .utils.metric import glue_metrics
from .model import model_type
from .utils import configure_cudnn
from .utils.dataset import make_testloader

import torch
from torch.utils.data import Dataset, DataLoader

def get_arg_parser():

    parser = argparse.ArgumentParser(
        description="Code for evaluating any PLMs provided by HuggingFace on Benchmark tasks")
    
    parser.add_argument('--ckpt-path', type=str, default='', action='')
    parser.add_argument('--data-type', '-d', type=str, default='glue')
    parser.add_argument('--task-type', '-t', type=str, default='cola')
    parser.add_argument('--modeling-type', type=str, default='single')
    parser.add_argument('--is-swa', '-s', type=bool, default=False)

    return parser

def evaluate():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = get_arg_parser()
    args = parser.parse_args()
    default_path = ['./finetuning', f'{args.data_type}-base', f'{args.task_type}.yaml']
    config_path = os.path.join(*default_path)

    swa_path = None
    if args.is_swa:
        swa_path = ['./finetuning', f'{args.data_type}-swa', f'{args.task_type}.yaml']
        swa_path = os.path.join(*swa_path)

    config = chaeyun_load(args.config_path, swa_path)
    configure_cudnn(True)

    print("Loading Model...")

    model = model_type[f'{config.dataset.type}_{args.type}'](config)
    state_dict = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(detach_module(state_dict))
    model.to(device)
    
    print("Loading Test dataset...")

    data = datasets.load_dataset('glue', config.dataset.name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    padding = 'max_length' if config.dataset.pad_to_max_length else False
    test_loader = make_testloader(config.dataset.batch_size,
                                  data, config.common.num_worker,
                                  config.dataset.name, tokenizer, 
                                  padding, config.dataset.max_sequence_length)
    metric = glue_metrics(config.dataset.name)
    _, acc, metric = validate(model, None,
                           metric, test_loader, 
                           config.dataset.name, device)

if __name__ == '__main__':
    evaluate()