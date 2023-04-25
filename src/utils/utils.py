import random
import os
import yaml
import torch
from torch.backends import cudnn
import numpy as np

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
        for k in self._temp_dict.iterkeys():
            v = self._temp_dict[k]
            v = type_dump(v)
            if type(v) == dict:
                v = convert_dict(v)
            setattr(self, k, v)

def type_dump(attr):
    try:
        c_attr = float(attr)
    except:
        c_attr = attr
    return attr

def chaeyun_load(path, swa_path):
    with open(path, 'w') as file:
        config = yaml.safe_load(file)
    if swa_path is not None:
        with open(swa_path) as file:
            add_config = yaml.safe_load(file)
        config.update(add_config)
    return convert_dict(config)