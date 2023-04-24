import os
import argparse
import torch
import sys
import tabulate
import numpy as np
import datasets
from .utils.metric import metrics
from .train import validate
from .model import RobertaGLUE
from .utils import set_seed, configure_cudnn
from transformers import RobertaTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader, random_split
from .utils import seed_worker
from collections import OrderedDict

from torch import cuda
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

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
    
    train_loader = DataLoader(
            train_data,
            batch_size = batch_size,
            num_workers = num_workers,
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

def get_arg_parser():

    parser = argparse.ArgumentParser(description="GLUE Roberta training")

    # Default settings
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    parser.add_argument('--swa_ckpt', type=str, default='')
    parser.add_argument('--base_ckpt1', type=str, default='')
    parser.add_argument('--base_ckpt2', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:32460', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')

    # Data settings
    parser.add_argument('--data_path', '-d', type=str, default='')
    parser.add_argument('--task', '-t', type=str, default='mnli')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--pad_to_max_length', type=bool, default=True)
    parser.add_argument('--max_seq_length', default=400, type=int)

    # Plot settings
    parser.add_argument('--grid_points', type=int, default=21, metavar='N',
                        help='number of points in the grid (default: 21)')
    parser.add_argument('--margin_left', type=float, default=0.2, metavar='M',
                        help='left margin (default: 0.2)')
    parser.add_argument('--margin_right', type=float, default=0.2, metavar='M',
                        help='right margin (default: 0.2)')
    parser.add_argument('--margin_bottom', type=float, default=0.2, metavar='M',
                        help='bottom margin (default: 0.)')
    parser.add_argument('--margin_top', type=float, default=0.2, metavar='M',
                        help='top margin (default: 0.2)')

    # Hyperparameter settings
    parser.add_argument('--model_type', '-m', type=str, default='roberta-base')

    return parser

def main():

    parser = get_arg_parser()
    args = parser.parse_args()

    n_gpus_per_node = cuda.device_count()
    args.world_size = n_gpus_per_node * args.world_size
    
    mp.spawn(main_worker, nprocs=n_gpus_per_node, args=(n_gpus_per_node, args))


def main_worker(gpu, n_gpus_per_node, args):

    set_seed(args.seed)
    configure_cudnn(False)

    args.gpu = gpu
    n_gpus_per_node = cuda.device_count()

    args.rank = args.rank * n_gpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print("Preparing data...")

    args.batch_size = int(args.batch_size / n_gpus_per_node)
    args.num_worker = int(args.num_workers / n_gpus_per_node)

    data = datasets.load_dataset('glue', args.task)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_len = args.max_seq_length
    padding = 'max_length' if args.pad_to_max_length else False
    train_loader, valid_loader, test_loader = make_dataloader(
        args.batch_size,
        data, args.num_workers,
        args.task, tokenizer, 
        padding, max_len, args.seed)
    
    print("Loading models...")

    def load_model(args):
        config = {}
        config['model_type'] = args.model_type
        config['classifier_dropout'] = 0.1
        config['num_labels'] = args.num_classes
        config['hidden_size'] = 768
        return RobertaGLUE(config)

    def load_trained_model(ckpts, args):
        models = []
        for ckpt in ckpts:
            temp = load_model(args)
            try:
                temp.load_state_dict(torch.load(ckpt, map_location="cuda:0")['model_state_dict'])
            except:
                new_state_dict = OrderedDict()
                state_dict = torch.load(ckpt, map_location="cuda:0")['model_state_dict']
                for k, v in state_dict.items():
                    if k == 'n_averaged':
                        continue
                    if 'module' in k:
                        name = k[14:]
                        new_state_dict[name] = v
                    else:
                        name = k
                        new_state_dict[name] = v
                temp.load_state_dict(new_state_dict)

            models.append(temp)
        return models
    
    base_model = load_model(args)
    cuda.set_device(args.gpu)
    base_model.cuda(args.gpu)
    base_model = torch.nn.parallel.DistributedDataParallel(base_model, device_ids=[args.gpu])
    trained_models = load_trained_model([args.base_ckpt1, args.base_ckpt2, args.swa_ckpt], args)

    criterion = torch.nn.MSELoss() if args.task == 'stsb' else torch.nn.CrossEntropyLoss()
    metric = metrics(args.task)

    def get_xy(point, origin, vector_x, vector_y):
        return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

    w = list()
    curve_parameters = list(model.parameters() for model in trained_models)
    for model in trained_models:
        w.append(
            np.concatenate([
            p.data.cpu().numpy().ravel() for p in model.parameters()
            ])
        )

    print('Weight space dimensionality: %d' % w[0].shape[0])
    
    u = w[2] - w[0]
    dx = np.linalg.norm(u)
    u /= dx
    
    v = w[1] - w[0]
    v -= np.dot(u, v) * u
    dy = np.linalg.norm(v)
    v /= dy

    bend_coordinates = np.stack(get_xy(p, w[0], u, v) for p in w)
    
    '''
    ts = np.linspace(0.0, 1.0, args.curve_points)
    curve_coordinates = []
    for t in np.linspace(0.0, 1.0, args.curve_points):
        weights = model.weights(torch.Tensor([t]).cuda())
        curve_coordinates.append(get_xy(weights, w[0], u, v))
    curve_coordinates = np.stack(curve_coordinates)
    '''
    #sys.exit()

    G = args.grid_points
    alphas = np.linspace(0.0 - args.margin_left, 1.0 + args.margin_right, G)
    betas  = np.linspace(0.0 - args.margin_bottom, 1.0 + args.margin_top, G)

    tr_loss = np.zeros((G, G))
    tr_metric = np.zeros((G, G))
    tr_acc = np.zeros((G, G))
    tr_err = np.zeros((G, G))

    te_loss = np.zeros((G, G))
    te_metric = np.zeros((G, G))
    te_acc = np.zeros((G, G))
    te_err = np.zeros((G, G))

    grid = np.zeros((G, G, 2))

    columns = ['X', 'Y', 'Train loss', 'Train acc', 'Train error (%)', 'Test acc', 'Test error (%)']

    print("Start contouring...")
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            p = w[0] + alpha * dx * u + beta * dy * v

            offset = 0
            for parameter in base_model.parameters():
                size = np.prod(parameter.size())
                value = p[offset:offset+size].reshape(parameter.size())
                parameter.data.copy_(torch.from_numpy(value))
                offset += size
            
            tr_loss_v, tr_acc_v, tr_metric_v = validate(base_model,
                                                        criterion,
                                                        metric,
                                                        train_loader,
                                                        args.task,
                                                        'cuda')
            te_loss_v, te_acc_v, te_metric_v = validate(base_model,
                                                        criterion, 
                                                        metric, 
                                                        test_loader, 
                                                        args.task,
                                                        'cuda')
            
            c = get_xy(p, w[0], u, v)
            grid[i, j] = [alpha * dx, beta * dy]

            tr_loss[i, j] = tr_loss_v
            tr_metric[i, j] = tr_metric_v
            tr_acc[i, j] = tr_acc_v
            tr_err[i, j] = 100.0 - tr_acc[i, j]

            te_loss[i, j] = te_loss_v
            te_metric[i, j] = te_metric_v
            te_acc[i, j] = te_acc_v
            te_err[i, j] = 100.0 - te_acc[i, j]

            values = [
                grid[i, j, 0], grid[i, j, 1], tr_loss[i, j], tr_metric[i, j], tr_err[i, j],
                te_metric[i, j], te_err[i, j]
            ]
            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
            if j == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table)
    
    np.savez(
        os.path.join('./ckpt', f'{args.task}_plane_21.npz'),
        bend_coordinates=bend_coordinates,
        alphas=alphas,
        betas=betas,
        grid=grid,
        tr_loss=tr_loss,
        tr_acc=tr_acc,
        tr_metric=tr_metric,
        tr_err=tr_err,
        te_loss=te_loss,
        te_acc=te_acc,
        te_metric=te_metric,
        te_err=te_err
    )

if __name__ == '__main__':
    main()