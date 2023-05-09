import os
import yaml
import random
import datetime

import torch
import numpy as np
from torch.backends import cudnn
from collections import OrderedDict
from tabulate import tabulate

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(seed):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)

def configure_cudnn(debug):
    cudnn.enabled = True
    cudnn.benchmark = True
    if debug:
        cudnn.deterministic = True
        cudnn.benchmark = False

class convert_dict:
    def __init__(self, temp_dict):
        self._temp_dict = temp_dict
        self._parse()
    
    def _parse(self):
        for k in self._temp_dict.keys():
            v = self._temp_dict[k]
            if type(v) == dict:
                v = convert_dict(v)
            setattr(self, k, v)

def chaeyun_load(path, swa_path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    if swa_path is not None:
        with open(swa_path, 'r') as f:
           add_config = yaml.safe_load(f)
        config.update(add_config)
    return convert_dict(config)

def detach_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict['model_state_dict'].items():
         if k == 'n_averaged':
              continue
         if 'module' in k:
              name = k[14:]
              new_state_dict[name] = v
         else:
              name = k
              new_state_dict[name] = v
    return new_state_dict

class chaeyun_average(object):
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 

class chaeyun_logs(object):
    def __init__(self):
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.train_time = []

    def update(self, tl, vl, ta, va, tt):
        self.train_loss.append(tl)
        self.valid_loss.append(vl)
        self.train_acc.append(ta)
        self.valid_loss.append(va)
        self.train_time.append(tt)

    def summary(self):
        my_headr = ['best_epoch','best_acc','average_time']
        best_epoch = self.valid_acc.index(max(self.valid_acc)) + 1
        total_time = datetime.timedelta(seconds=np.mean(self.train_time))
        my_table = [[best_epoch, self.valid_acc[best_epoch], total_time]] 
        print("===== Training Results =====")
        print(tabulate(my_table, headers=my_headr, tablefmt='orgtbl'))

    def result(self):
        return {
            'train_loss': self.train_loss,
            'valid_loss': self.valid_loss,
            'train_acc': self.train_acc,
            'valid_acc': self.valid_acc,
            'train_time': self.train_time
        }