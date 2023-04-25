import torch
import datasets
from transformers import RobertaTokenizer
from .train import validate
from .utils.metric import metrics
from .model import RobertaGLUE
from collections import OrderedDict
from .utils import set_seed, configure_cudnn

import torch
from torch.utils.data import Dataset, DataLoader

def evaluate():

    set_seed(42)
    configure_cudnn(True)

    print("Loading Model...")
    config = {}
    config['model_type'] = 'roberta-base'
    config['classifier_dropout'] = 0.1
    config['num_labels'] = 2
    config['hidden_size'] = 768

    model = RobertaGLUE(config)

    state_dict = torch.load('/mnt/home/chaeyun-jang/roberta/complete_for_vision/cola/cola_2e-05_42.pt', map_location='cpu')

    from collections import OrderedDict

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
    
    #model.load_state_dict(state_dict['model_state_dict'])

    model.load_state_dict(new_state_dict)
    print("Loading Test dataset...")
    data = datasets.load_dataset('glue', 'cola')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_len = 200
    padding = 'max_length'
    _, valid_loader, _ = make_dataloader(
        32,
        data, 1,
        'cola', tokenizer, 
        padding, max_len)
    metric = metrics('cola')
    _, acc, add = validate(model,
             None,
             metric,
             valid_loader, 
             'cola',
             'cpu')
    print(acc)
    print(add)

if __name__ == '__main__':
    evaluate()