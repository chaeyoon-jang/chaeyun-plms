import os
import os.path as p

import time
import datetime

import torch
from torch import nn
from torch import cuda
from functools import partial

import json
from tqdm import tqdm

from .utils import chaeyun_average, chaeyun_logs

def save_model(model, epoch, acc, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'best_acc': acc
    }, path)

def save_log(logs, path):
    with open(path, 'w') as f:
        json.dump(logs, f)

def validate(
        model, criterion, metric, valid_loader, device):
    
    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        valid_acc = 0.0
        valid_metric = 0.0
        for data in valid_loader:

            input_ids, attention_mask, targets = data
            input_ids = input_ids.cuda(device)
            attention_mask = attention_mask.cuda(device)
            targets = targets.cuda(device)

            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask)

            if criterion:
                loss = criterion(outputs.logits, targets) 
                valid_loss += loss.item()
            
            acc, _metric = metric.calculate(outputs.logits, targets)

            valid_acc += acc
            valid_metric += _metric
    
    return valid_loss / len(valid_loader), valid_acc / len(valid_loader), valid_metric / len(valid_loader)

def train(
        model, criterion, metric, optimizers, data_loader, config):
    
    device = 'cuda' if config.common.gpu > 0 else 'cpu'
    ckpt_path = p.join(config.common.save_dir,\
                f'./ckpt/{config.dataset.name}_{config.optimizer.learning_rate}_{config.common.seed}.pt')
    log_path = p.join(config.common.save_dir,\
                f'./log/{config.dataset.name}_{config.optimizer.learning_rate}_{config.common.seed}.json')

    train_loader, valid_loader = data_loader
    optimizer, lr_scheduler = optimizers

    best_acc = 0.0
    best_metric = 0.0
    logs = chaeyun_logs()

    for epoch in range(config.common.n_epochs):

        train_loss = chaeyun_average()
        train_acc = chaeyun_average()

        model.train()
        epoch_start = time.time()
        total = len(train_loader)

        with tqdm(total=total) as pbar:
            for data in train_loader:
                start = time.time()
                
                input_ids, attention_mask, targets = data
                input_ids = input_ids.cuda(config.common.gpu)
                attention_mask = attention_mask.cuda(config.common.gpu)
                targets = targets.cuda(config.common.gpu)

                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask)
                
                with torch.autocast('cuda'):
                    loss = criterion(outputs.logits, targets)
                    acc, _ = metric.calculate(outputs.logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if config.optimizer.name != 'sgd':
                    lr_scheduler.step()

                train_loss.update(loss.item())
                train_acc.update(acc)
                pbar.update(1)

        epoch_train_loss = train_loss.avg
        epoch_train_acc = train_acc.avg

        elapse_time = time.time() - epoch_start

        epoch_valid_loss, epoch_valid_acc, epoch_valid_add = validate(model, criterion, 
                                                                      metric, valid_loader, config.common.gpu)
        logs.update(epoch_train_loss, epoch_valid_loss, 
                    epoch_train_acc, epoch_valid_acc, 
                    elapse_time)
        
        elapse_time = datetime.timedelta(seconds=elapse_time)
        print(f"Epoch: {epoch+1} | train loss: {epoch_train_loss:.4f} | train acc: {epoch_train_acc:.4f}% | time: {elapse_time}")
        print(f"Epoch: {epoch+1} | valid loss: {epoch_valid_loss:.4f} | valid acc: {epoch_valid_acc:.4f}% | metric: {epoch_valid_add:.4f}%")
        
        if epoch_valid_acc > best_acc:
            best_acc = epoch_valid_acc
            save_model(model, epoch, best_acc, ckpt_path)
    
    logs.summary()
    save_log(logs.result(), log_path)

def train_swa(
        model, criterion, metric, optimizers, data_loader, config):

    device = 'cuda' if config.common.gpu > 0 else 'cpu'
    ckpt_path = p.join(config.common.save_dir,\
                f'./ckpt/{config.dataset.name}_{config.optimizer.learning_rate}_{config.common.seed}_swa.pt')
    log_path = p.join(config.common.save_dir,\
                f'./log/{config.dataset.name}_{config.common.learning_rate}_{config.common.seed}_swa.json')

    train_loader, valid_loader = data_loader
    (optimizer, lr_scheduler), swa_optimizers = optimizers
    swa_scheduler, swa_model = swa_optimizers

    best_acc = 0.0
    best_metric = 0.0
    logs = chaeyun_logs()

    for epoch in range(config.common.n_epochs):

        train_loss = chaeyun_average()
        train_acc = chaeyun_average()

        model.train()
        epoch_start = time.time()
        total = len(train_loader)

        with tqdm(total=total) as pbar:
            for data in train_loader:
                start = time.time()

                input_ids, attention_mask, targets = data
                input_ids = input_ids.cuda(config.common.gpu)
                attention_mask = attention_mask.cuda(config.common.gpu)
                targets = targets.cuda(config.common.gpu)

                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask)

                with torch.autocast('cuda'):
                    loss = criterion(outputs.logits, targets)
                    acc, _ = metric.calculate(outputs.logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                if (epoch <= config.swa.start_epoch) & (config.optimizer.name != 'sgd') :      
                    lr_scheduler.step()

                train_loss.update(loss.item())
                train_acc.update(acc)
                pbar.update(1)

        if epoch > config.swa.start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step() 

        epoch_train_loss = train_loss.avg
        epoch_train_acc = train_acc.avg

        elapse_time = time.time() - epoch_start
        
        validate_fn = partial(validate, criterion=criterion, 
                              metric=metric, valid_loader=valid_loader, device=config.common.gpu)
        
        epoch_valid_loss, epoch_valid_acc, epoch_valid_metric = validate_fn(model=swa_model)\
            if epoch > config.swa.start_epoch else validate_fn(model=model)
        
        logs.update(epoch_train_loss, epoch_valid_loss,
                    epoch_train_acc, epoch_valid_acc,
                    elapse_time)
        
        elapse_time = datetime.timedelta(seconds=elapse_time)
        print(f"Epoch: {epoch+1} | train loss: {epoch_train_loss:.4f} | train acc: {epoch_train_acc:.4f}% | time: {elapse_time}")
        print(f"Epoch: {epoch+1} | valid loss: {epoch_valid_loss:.4f} | valid acc: {epoch_valid_acc:.4f}% | metric: {epoch_valid_metric:.4f}%")

        if epoch_valid_acc > best_acc:
            best_acc = epoch_valid_acc
            save_model(swa_model, epoch, best_acc, ckpt_path)
    
    logs.summary()
    save_log(logs.result(), log_path)