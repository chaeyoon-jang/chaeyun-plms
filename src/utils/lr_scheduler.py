import torch
from transformers.optimization import AdamW
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import SWALR
from transformers import get_linear_schedule_with_warmup

def get_optimizer(model, args):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
    ]

    if args.optimizer == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            eps=args.adam_epsilon,
                            betas=(0.9,0.98))

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.total_steps)
    
    elif args.optimizer == "sgd":
        optimizer = SGD(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        momentum=args.optim_momentum)
        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.total_steps)
        
    if args.is_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_learning_rate)
        swa_optim = (swa_model, swa_scheduler)
    
    else:
        swa_optim = None

    return optimizer, scheduler, swa_optim