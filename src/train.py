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
    ckpt_path = p.join(config.common.save_dir,\
                f'./ckpt/{config.dataset.name}_{config.optimizer.learning_rate}_{config.common.seed}.pt')
    log_path = p.join(config.common.save_dir,\
                f'./log/{config.dataset.name}_{config.optimizer.learning_rate}_{config.common.seed}.json')

    train_loader, valid_loader = data_loader
    (optimizer, lr_scheduler), _ = optimizers

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
            for batch_idx, data in enumerate(train_loader):
                start = time.time()

                input_ids = data['input_ids'].cuda(config.common.gpu)
                attention_mask = data['attention_mask'].cuda(config.common.gpu)
                token_type_ids = data['token_type_ids'].cuda(config.common.gpu)
                targets = data['label'].cuda(config.common.gpu)

                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
                
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
                                                                      metric, valid_loader, 
                                                                      config.dataset.name, config.common.gpu)
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
            for batch_idx, data in enumerate(train_loader):
                start = time.time()

                input_ids = data['input_ids'].cuda(config.common.gpu)
                attention_mask = data['attention_mask'].cuda(config.common.gpu)
                token_type_ids = data['token_type_ids'].cuda(config.common.gpu)
                targets = data['label'].cuda(config.common.gpu)

                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

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
                              metric=metric, valid_loader=valid_loader, 
                              task=config.dataset.name, device=config.common.gpu)
        
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
                
def train_multi(
        model, criterion, metric, optimizers, data_loader, config):
    
    tasks = config.dataset.task 
    optimizer_ = config.optimizer
    common = config.common
    
    device = 'cuda' if common.gpu > 0 else 'cpu'
    ckpt_path = p.join(common.save_dir,\
                f'./ckpt/multitask_{common.learning_rate}_{common.seed}.pt')
    log_path = p.join(common.save_dir,\
                f'./log/multitask_{common.learning_rate}_{common.seed}.json')

    train_loader, valid_loader = data_loader
    task_names =  list(valid_loader.keys())
    optimizer, lr_scheduler = optimizers

    best_acc = 0.0
    
    logs = dict()
    for name in range(len(task_names)):
        logs[name] = chaeyun_logs()

    for epoch in range(common.n_epochs):
        
        train_loss = dict()
        train_acc = dict()
        for name in task_names:
            train_loss[name] = chaeyun_average()
            train_acc[name] = chaeyun_average()

        model.train()
        epoch_start = time.time()
        total = len(train_loader)
        
        with tqdm(total=total) as pbar:
            # train_features, train_labels = next(iter(train_dataloader))
            # for _ in range(total):
            for data in train_loader:
                #data = next(iter(train_loader))
                start = time.time()

                input_ids = data['input_ids'].cuda(common.gpu)
                attention_mask = data['attention_mask'].cuda(common.gpu)
                premise_mask = data['premise_mask'].cuda(common.gpu)
                hyp_mask = data['hyp_mask'].cuda(common.gpu)
                targets = data['label'].to(torch.int64).cuda(common.gpu)
                task_name = data['task_name'][0]
                task = eval(f'tasks.{task_name}') 
                
                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                premise_mask=premise_mask,
                                hyp_mask = hyp_mask,
                                layer_type=task.layer_type,
                                enable_san=task.enable_san)
                
                with torch.autocast('cuda'):
                    if task.is_mse:
                        loss = criterion[0](outputs.logits, targets)
                    else:
                        loss = criterion[1](outputs.logits, targets)
                    acc, _ = metric.calculate(outputs.logits, targets, task.metric_type)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if optimizer_.name != 'sgd':
                    lr_scheduler.step()

                train_loss[task_name].update(loss.item())
                train_acc[task_name].update(acc)
                pbar.update(1)
        
        total_train_result = dict()
        for name in task_names:
            # epoch_train_loss, epoch_train_acc
            total_train_result[name] = [train_loss[name].avg, train_acc[name].avg]

        elapse_time = time.time() - epoch_start
        total_valid_result = dict()
        
        avg_train_acc  = 0.0
        avg_valid_acc  = 0.0
        
        for name in task_names:
            # epoch_valid_loss, epoch_valid_acc, epoch_valid_metric
            total_valid_result[name] = validate(model, criterion, metric, valid_loader[name], name, device)
            
            total_train_loss = total_train_result[name][0] 
            total_train_acc  = total_valid_result[name][0]
            total_valid_loss = total_train_result[name][1]
            total_valid_acc  = total_valid_result[name][1]
            total_valid_metric = total_valid_result[name][2]
            
            avg_train_acc  += total_train_acc
            avg_valid_acc  += total_valid_acc       
             
            logs[name].update(total_train_loss,
                              total_train_acc,
                              total_valid_loss,
                              total_valid_acc, 
                              elapse_time
                              )
            
            elapse_time = datetime.timedelta(seconds=elapse_time)
            print(f"================== {name} Data Results ==================")
            print(f"Epoch: {epoch+1} | train loss: {total_train_loss:.4f} | train acc: {total_train_acc:.4f}% | time: {elapse_time}")
            print(f"Epoch: {epoch+1} | valid loss: {total_valid_loss:.4f} | valid acc: {total_valid_acc:.4f}% | metric: {total_valid_metric:.4f}%")
            print(f"---------------------------------------------------------")
        
        avg_train_acc  /= len(task_names)
        avg_valid_acc  /= len(task_names)
        
        print(f"================== {name} Data Results ==================")
        print(f"Epoch: {epoch+1} | train acc: {avg_train_acc:.4f} | valid acc: {avg_valid_acc:.4f}%")
        print(f"---------------------------------------------------------")
                                                                                                                    
        if avg_valid_acc > best_acc:
            best_acc = avg_valid_acc
            save_model(model, epoch, best_acc, ckpt_path)
    
    total_logs = dict()
    for name in task_names:
        print(f"================== {name} Data Summary ==================")
        logs[name].summary()
        print(f"---------------------------------------------------------")
        total_logs[name] = logs[name].result()
    
    save_log(total_logs, log_path)