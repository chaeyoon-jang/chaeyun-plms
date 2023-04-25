import os
import os.path as p

import argparse
import datasets
from datetime import datetime

import torch
from torch import cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, distributed
from transformers import DataCollatorWithPadding
from transformers import RobertaTokenizer, AutoConfig

from .model import SingleTaskClassificationModel, MultiTaskClassificationModel
from .utils import set_seed, configure_cudnn, chaeyun_load
from .utils.metric import metrics
from .utils.dataset import make_dataloader
from .utils.lr_scheduler import get_optimizer
from .train import train, train_swa, validate

import warnings
warnings.filterwarnings('ignore')

def get_arg_parser():

    parser = argparse.ArgumentParser(
        description="Code for fine-tuning any PLMs provided by HuggingFace on Benchmark tasks")
    
    parser.add_argument('--data-type', '-d', type=str, default='glue')
    parser.add_argument('--task-type', '-t', type=str, default='cola')
    parser.add_argument('--is-swa', '-s', type=bool, default=False)
    # TO DO: debug code
    parser.set_defaults(debug=False)

    return parser

def main():

    parser = get_arg_parser()
    args = parser.parse_args()

    default_path = ['./finetuning', f'{args.data_type}-base', f'{args.task_type}.yaml']
    config_path = os.path.join(*default_path)

    swa_path = None
    if args.is_swa:
        swa_path = ['./finetuning', f'{args.data_type}-swa', f'{args.task_type}.yaml']
        swa_path = os.path.join(*swa_path)

    config = chaeyun_load(args.config_path, swa_path)
    
    set_seed(config.common.seed)
    configure_cudnn(False)

    n_gpus_per_node = cuda.device_count()

    config.common.world_size = n_gpus_per_node * config.common.world_size
    mp.spawn(main_worker, nprocs=n_gpus_per_node, args=(n_gpus_per_node, config))

def main_worker(gpu, n_gpus_per_node, config):

    set_seed(config.common.seed)
    configure_cudnn(False)
    
    config.common.gpu = gpu
    n_gpus_per_node = cuda.device_count()

    print("Use GPU: {} for training...".format(config.common.gpu))

    config.common.rank = config.common.rank * n_gpus_per_node + gpu
    dist.init_process_group(backend=config.common.dist_backend, init_method=config.common.dist_url,
                            world_size=config.common.world_size, rank=config.common.rank)
    print("Making model...")
    model = SingleTaskClassificationModel(config)
    cuda.set_device(config.common.gpu)
    model.cuda(config.common.gpu)

    config.dataset.batch_size = int(config.dataset.batch_size / n_gpus_per_node)
    config.common.num_worker = int(config.common.num_workers / n_gpus_per_node)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.common.gpu])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters of model is {}...".format(num_params))

    print("Preparing data...")
    data = datasets.load_dataset('glue', config.dataset.name)
    tokenizer = RobertaTokenizer.from_pretrained(config.model.name)
    max_len = config.dataset.max_seq_length
    padding = 'max_length' if config.dataset.pad_to_max_length else False
    train_loader, valid_loader, test_loader = make_dataloader(
        config.dataset.batch_size,
        data, config.common.num_workers,
        config.dataset.name, tokenizer, 
        padding, max_len, config.common.seed)

    print("Start training...")
    os.makedirs(config.common.save_dir, exist_ok=True)
    metric = metrics(config.dataset.name)
    criterion = torch.nn.CrossEntropyLoss()
    if config.dataset.name == "stsb":
        criterion = torch.nn.MSELoss()
    config.lr_scheduler.total_steps = int(len(train_loader) * config.common.n_epochs)
    config.lr_scheduler.warmup_steps = int(config.lr_scheduler.total_steps * config.lr_scheduler.warmup_ratio) 
    optimizer, lr_scheduler, swa_optim = get_optimizer(model, config)

    if args.is_swa:
        train_swa(model=model, 
          criterion=criterion, 
          metric=metric, 
          optimizer=optimizer, 
          lr_scheduler=lr_scheduler,
          swa_optim=swa_optim,
          swa_epoch=args.swa_num_pt_epochs,
          train_loader=train_loader, 
          valid_loader=valid_loader,
          epochs=args.n_epochs,
          task=args.task,
          device='cuda',
          ckpt_path=args.save_dir,
          args=args)
    
    else:
        train(model=model, 
            criterion=criterion, 
            metric=metric, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler,
            train_loader=train_loader, 
            valid_loader=valid_loader,
            epochs=args.n_epochs,
            task=args.task,
            device='cuda',
            ckpt_path=args.save_dir,
            args=args)

if __name__ == '__main__':
    main()