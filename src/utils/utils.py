import random
import os
import yaml
import torch
import numpy as np
from torch.backends import cudnn
from collections import OrderedDict

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

def type_dump(attr):
    try:
        c_attr = float(attr)
    except:
        c_attr = attr
    return c_attr

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