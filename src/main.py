import os
import os.path as p

import torch
import argparse
import datasets
from datetime import datetime
from torch import cuda
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from .model import RobertaGLUE
from .utils import set_seed, configure_cudnn
from .utils.dataset import make_dataloader
from .utils.lr_scheduler import get_optimizer
from .train import train, train_swa, validate
from .utils.metric import metrics
from transformers import RobertaTokenizer, AutoConfig

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import warnings
warnings.filterwarnings('ignore')

def get_arg_parser():

    parser = argparse.ArgumentParser(description="GLUE Roberta training")

    # Default settings
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:32460', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--save_dir', type=str, default="./ckpt")

    # Data settings
    parser.add_argument('--data_path', '-d', type=str, default='')
    parser.add_argument('--task', '-t', type=str, default='mnli')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--pad_to_max_length', type=bool, default=True)
    parser.add_argument('--max_seq_length', default=400, type=int)

    # Hyperparameter settings
    parser.add_argument('--model_type', '-m', type=str, default='roberta-base')
    parser.add_argument('--n_epochs', '-e', type=int, default=10)
    parser.add_argument('--learning_rate', default=1e-05, type=float)
    parser.add_argument('--adam_epsilon', type=float, default=1e-06)
    parser.add_argument('--adam_bias', type=bool, default=True)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--total_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--optim_momentum', type=float, default=0.9)

    # Stochastic weights averaging settings
    parser.add_argument('--is_swa', type=bool, default=False)
    parser.add_argument('--swa_num_pt_epochs',type=int, default=3)
    parser.add_argument('--swa_learning_rate', type=float, default=1e-05)

    return parser


def main():

    parser = get_arg_parser()
    args = parser.parse_args()
    
    set_seed(args.seed)
    configure_cudnn(False)

    n_gpus_per_node = cuda.device_count()

    args.world_size = n_gpus_per_node * args.world_size
    
    mp.spawn(main_worker, nprocs=n_gpus_per_node, args=(n_gpus_per_node, args))

def main_worker(gpu, n_gpus_per_node, args):

    set_seed(args.seed)
    configure_cudnn(False)
    
    args.gpu = gpu
    n_gpus_per_node = cuda.device_count()

    print("Use GPU: {} for training...".format(args.gpu))

    args.rank = args.rank * n_gpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    print("Making model...")
    config = {}
    config['model_type'] = args.model_type
    config['classifier_dropout'] = 0.1
    config['num_labels'] = args.num_classes
    config['hidden_size'] = 768

    model = RobertaGLUE(config)
    cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    args.batch_size = int(args.batch_size / n_gpus_per_node)
    args.num_worker = int(args.num_workers / n_gpus_per_node)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters of model is {}...".format(num_params))

    print("Preparing data...")
    data = datasets.load_dataset('glue', args.task)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_len = args.max_seq_length
    padding = 'max_length' if args.pad_to_max_length else False
    train_loader, valid_loader, test_loader = make_dataloader(
        args.batch_size,
        data, args.num_workers,
        args.task, tokenizer, 
        padding, max_len, args.seed)

    print("Start training...")
    os.makedirs(args.save_dir, exist_ok=True)
    metric = metrics(args.task)
    criterion = torch.nn.CrossEntropyLoss()
    if args.task == "stsb":
        criterion = torch.nn.MSELoss()
    args.total_steps = int(len(train_loader) * 20)
    args.warmup_steps = int(args.total_steps * 0.1) 
    optimizer, lr_scheduler, swa_optim = get_optimizer(model, args)

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