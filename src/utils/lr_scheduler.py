import torch
from torch.optim import SGD
from torch.optim.swa_utils import SWALR
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup

def get_optimizer(model, config):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.optimizer.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
    ]

    if config.optimizer.name == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters,
                            lr=config.optimizer.learning_rate,
                            eps=config.optimizer.adam_epsilon,
                            betas=(config.optimizer.adam_betas[0], config.optmizer.adam_betas[0]))

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.lr_scheduler.warmup_steps,
                num_training_steps=config.lr_sheduler.total_steps)
    
    elif config.optimizer.name == "sgd":
        optimizer = SGD(optimizer_grouped_parameters,
                        lr=config.optimizer.learning_rate,
                        momentum=config.optmizer.optim_momentum)
        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.lr_scheduler.warmup_steps,
                num_training_steps=config.lr_scheduler.total_steps)
        
    if config.swa.is_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=config.swa.learning_rate)
        swa_optim = (swa_model, swa_scheduler)
    
    else:
        swa_optim = None

    return ((optimizer, scheduler), swa_optim)