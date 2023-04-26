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

def validate(
        model, criterion, metric, valid_loader, task, device):
    
    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        valid_acc = 0.0
        valid_metric = 0.0
        for batch_idx, data in enumerate(valid_loader):

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            targets = data['label'].to(device)

            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

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

    train_loader, valid_loader = data_loader
    (optimizer, lr_scheduler), _ = optimizers

    best_acc = 0.0
    best_metric = 0.0

    for epoch in range(config.common.n_epochs):

        train_loss = 0.0
        train_acc = 0.0

        model.train()
        epoch_start = time.time()
        total = len(train_loader)

        with tqdm(total=total) as pbar:
            for batch_idx, data in enumerate(train_loader):
                start = time.time()

                input_ids = data['input_ids'].cuda(config.device)
                attention_mask = data['attention_mask'].cuda(device)
                token_type_ids = data['token_type_ids'].cuda(device)
                targets = data['label'].cuda(device)

                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
                
                with torch.autocast('cuda'):
                    loss = criterion(outputs['logits'], targets)
                    acc, _ = metric.calculate(outputs['logits'], targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if config.optimizer.name != 'sgd':
                    lr_scheduler.step()

                train_loss += loss.item()
                train_acc += acc
                pbar.update(1)

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_acc / len(train_loader)

        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)

        print(f"Epoch: {epoch+1} | train loss: {epoch_train_loss:.4f} |\
               train acc: {epoch_train_acc:.4f}% | time: {elapse_time}")
        epoch_valid_loss, epoch_valid_acc, epoch_valid_add = validate(model, criterion, 
                                                                      metric, valid_loader, 
                                                                      config.dataset.name, device)

        print(f"Epoch: {epoch+1} | valid loss: {epoch_valid_loss:.4f} |\
               valid acc: {epoch_valid_acc:.4f}% | metric: {epoch_valid_add:.4f}%")
        
        if epoch_valid_acc > best_acc:
                best_add = epoch_valid_add * 100
                best_acc = epoch_valid_acc
                torch.save({
                    'epoch':epoch,
                    'model_state_dict': model.module.state_dict(),
                    'best_metric':best_add,
                }, p.join(config.common.save_dir, 
                          f'{config.dataset.name}_{config.optimizer.learning_rate}_{config.common.seed}.pt'))

def train_swa(
        model, criterion, metric, optimizers, data_loader, config):

    device = 'cuda' if config.common.gpu > 0 else 'cpu'

    train_loader, valid_loader = data_loader
    (optimizer, lr_scheduler), swa_optimizers = optimizers
    swa_scheduler, swa_model = swa_optimizers

    best_acc = 0.0
    best_metric = 0.0

    for epoch in range(config.common.n_epochs):

        train_loss = 0.0
        train_acc = 0.0

        model.train()
        epoch_start = time.time()
        total = len(train_loader)

        with tqdm(total=total) as pbar:
            for batch_idx, data in enumerate(train_loader):
                start = time.time()

                input_ids = data['input_ids'].cuda(device)
                attention_mask = data['attention_mask'].cuda(device)
                token_type_ids = data['token_type_ids'].cuda(device)
                targets = data['label'].cuda(device)

                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

                with torch.autocast('cuda'):
                    loss = criterion(outputs['logits'], targets)
                    acc, _ = metric.calculate(outputs['logits'], targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                if (epoch <= config.swa.start_epoch) & (config.optimizer.name != 'sgd') :      
                    lr_scheduler.step()

                train_loss += loss.item()
                train_acc += acc
                pbar.update(1)

        if epoch > config.swa.start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step() 

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_acc / len(train_loader)

        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)

        print(f"Epoch: {epoch+1} | train loss: {epoch_train_loss:.4f} |\
               train acc: {epoch_train_acc:.4f}% | time: {elapse_time}")
        
        validate_fn = partial(validate, criterion=criterion, 
                              metric=metric, valid_loader=valid_loader, 
                              task=config.dataset.name, device=device)
        
        epoch_valid_loss, epoch_valid_acc, epoch_valid_metric = validate_fn(model=swa_model)\
            if epoch > config.swa.start_epoch else validate_fn(model=model)
        
        print(f"Epoch: {epoch+1} | valid loss: {epoch_valid_loss:.4f} |\
               valid acc: {epoch_valid_acc:.4f}% | metric: {epoch_valid_metric:.4f}%")

        if epoch_valid_acc > best_acc:
            best_add = epoch_valid_metric * 100
            best_acc = epoch_valid_acc
            if epoch > config.swa.start_epoch:
                torch.optim.swa_utils.update_bn(train_loader, swa_model)
                torch.save({
                    'epoch':epoch,
                    'model_state_dict': swa_model.state_dict(),
                    'best_acc':best_acc
                    }, p.join(config.common.save_path, f'{config.dataset.name}_{config.common.seed}_swa.pt'))
                
def train_multi(
        model, criterion, metric, optimizers, data_loader, config):
    
    device = 'cuda' if config.common.gpu > 0 else 'cpu'

    train_loader, valid_loader = data_loader
    optimizer, lr_scheduler = optimizers

    best_acc = 0.0
    best_metric = 0.0

    for epoch in range(config.common.n_epochs):

        train_loss = 0.0
        train_acc = 0.0
        model.train()
        epoch_start = time.time()
        total = len(train_loader)

        with tqdm(total=total) as pbar:
            for batch_idx, data in enumerate(train_loader):
                start = time.time()

                input_ids = data['input_ids'].cuda(config.device)
                attention_mask = data['attention_mask'].cuda(device)
                token_type_ids = data['token_type_ids'].cuda(device)
                targets = data['label'].to(torch.int64).cuda(device)

                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
                
                with torch.autocast('cuda'):
                    loss = criterion(outputs['logits'], targets)
                    acc, _ = metric.calculate(outputs['logits'], targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if config.optimizer.name != 'sgd':
                    lr_scheduler.step()

                train_loss += loss.item()
                train_acc += acc
                pbar.update(1)

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_acc / len(train_loader)

        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)

        print(f"Epoch: {epoch+1} | train loss: {epoch_train_loss:.4f} |\
               train acc: {epoch_train_acc:.4f}% | time: {elapse_time}")
        epoch_valid_loss, epoch_valid_acc, epoch_valid_add = validate(model, criterion, 
                                                                      metric, valid_loader, 
                                                                      config.dataset.name, device)

        print(f"Epoch: {epoch+1} | valid loss: {epoch_valid_loss:.4f} |\
               valid acc: {epoch_valid_acc:.4f}% | metric: {epoch_valid_add:.4f}%")
        
        if epoch_valid_acc > best_acc:
                best_add = epoch_valid_add * 100
                best_acc = epoch_valid_acc
                torch.save({
                    'epoch':epoch,
                    'model_state_dict': model.module.state_dict(),
                    'best_metric':best_add,
                }, p.join(config.common.save_dir, 
                          f'{config.dataset.name}_{config.optimizer.learning_rate}_{config.common.seed}.pt'))