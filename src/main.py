import os
import os.path as p

import argparse
import datasets
from functools import partial
from datetime import datetime

import torch
from torch import cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, distributed
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoConfig

from .model import model_type
from .utils import set_seed, configure_cudnn, chaeyun_load
from .utils.metric import glue_metrics, chaeyun_criterion
from .utils.dataset import make_glue_dataloader_v1,\
    make_glue_dataloader_v2,\
        make_multi_glue_dataloader

from .utils.lr_scheduler import get_optimizer
from .train import train, train_swa, validate

import warnings
import transformers

warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_arg_parser():

    parser = argparse.ArgumentParser(
        description="Code for fine-tuning any PLMs provided by HuggingFace on Benchmark tasks")
    
    parser.add_argument('--data-type', '-d', type=str, default='glue')
    parser.add_argument('--task-type', '-t', type=str, default='cola')
    parser.add_argument('--is-swa', '-s', type=bool, default=False)
    parser.add_argument('--DEBUG', dest='debug', action='store_true')
    parser.add_argument('--NO-DEBUG', dest='debug', action='store_false')
    parser.set_defaults(debug=False)

    return parser

def main():

    parser = get_arg_parser()
    args = parser.parse_args()

    default_path = ['./src/finetuning', f'{args.data_type}-base', f'{args.task_type}.yaml']
    config_path = os.path.join(*default_path)

    swa_path = None
    if args.is_swa:
        swa_path = ['./src/finetuning', f'{args.data_type}-swa', f'{args.task_type}.yaml']
        swa_path = os.path.join(*swa_path)

    config = chaeyun_load(config_path, swa_path)

    set_seed(config.common.seed)
    configure_cudnn(False)
    n_gpus_per_node = cuda.device_count()

    config.common.world_size = n_gpus_per_node * config.common.world_size
    mp.spawn(main_worker, nprocs=n_gpus_per_node, args=(n_gpus_per_node, config, args))

def main_worker(gpu, n_gpus_per_node, config, args):

    set_seed(config.common.seed)
    configure_cudnn(False)
    
    config.common.gpu = gpu

    print("Use GPU: {} for training...".format(config.common.gpu))
    config.common.rank = config.common.rank * n_gpus_per_node + gpu
    dist.init_process_group(backend=config.common.dist_backend, init_method=config.common.dist_url,
                            world_size=config.common.world_size, rank=config.common.rank)
    
    print("Making model...")
    training_type = 'multi' if args.data_type == 'multi-task' else 'single'
    model = model_type[f'{config.dataset.type}_{training_type}'](config)
    cuda.set_device(config.common.gpu)
    model.cuda(config.common.gpu)

    config.dataset.batch_size = int(config.dataset.batch_size / n_gpus_per_node)
    config.dataset.num_worker = int(config.dataset.num_workers / n_gpus_per_node)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.common.gpu])
    num_params = sum(params.numel() for params in model.parameters() if params.requires_grad)

    print("The number of parameters of model is {}...".format(num_params))

    print("Preparing data...")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    if args.data_type == 'glue':

        version = config.dataset.save_dir[-2:]; name_ = 'train'
        data_path = p.join(config.dataset.save_dir,f'{args.task_type}_{name_}_loader_{version}.pth')

        if not p.isfile(data_path):
            os.makedirs(config.dataset.save_dir, exist_ok=True)
            
            if args.task_type == 'multi-task':
                make_multi_glue_dataloader(tokenizer, config.dataset)
                metric = multi_glue_metrics()
            
            else:
                data = datasets.load_dataset('glue', config.dataset.name)
                exec(f'make_glue_dataloader_{version}(data, tokenizer, config.dataset)')
                metric = glue_metrics(config.dataset.name)
        
        data_loader = [torch.load(data_path) for name_ in ['train', 'valid']]

    data_loader = (data_loader[0], data_loader[1])

    print("Start training...")

    os.makedirs(p.join(config.common.save_dir,'./ckpt'), exist_ok=True)
    os.makedirs(p.join(config.common.save_dif,'./log'),  exist_ok=True)
    
    criterion = chaeyun_criterion(config.dataset.type)
    config.lr_scheduler.total_steps = int(len(data_loader[0]) * config.common.n_epochs)
    config.lr_scheduler.warmup_steps = int(config.lr_scheduler.total_steps * config.lr_scheduler.warmup_ratio) 
    optimizers = get_optimizer(model, config)

    train_module = partial(train_swa, model=model, criterion=criterion, metric=metric)\
        if args.is_swa else partial(train, model=model, criterion=criterion, metric=metric)
    
    train_module(optimizers=optimizers, data_loader=data_loader, config=config)

if __name__ == '__main__':
    main()