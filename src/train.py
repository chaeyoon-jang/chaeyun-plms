import os
import os.path as p

import time
import datetime

import torch
import json
from torch import cuda
from tqdm import tqdm

def validate(model,
             criterion,
             metric,
             valid_loader, 
             task,
             device):
    
    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        valid_acc = 0.0
        valid_add = 0.0
        for batch_idx, data in enumerate(valid_loader):

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            targets = data['label'].to(device)

            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            if criterion:
                if task == 'stsb':
                    targets = targets / 5.0
                loss = criterion(outputs['logits'], targets)
                valid_loss += loss.item()
            
            acc, add = metric.calculate(outputs['logits'], targets)

            valid_acc += acc
            valid_add += add
    
    return valid_loss / len(valid_loader), valid_acc / len(valid_loader), valid_add / len(valid_loader)


def train(model, 
          criterion, 
          metric, 
          optimizer, 
          lr_scheduler,
          train_loader, 
          valid_loader,
          epochs,
          task,
          device,
          ckpt_path,
          args):
    
    best_acc = 0.0
    best_add = 0.0
    train_results1 = []
    train_results2 = []
    results1 = []
    results2 = []
    for epoch in range(epochs):

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
                targets = data['label'].to(torch.int64).cuda(device)

                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
                
                with torch.autocast('cuda'):
                    if task == 'stsb':
                        outputs['logits'] = outputs['logits'].to(torch.float32)
                        targets = targets / 5.0
                        targets = targets.to(torch.float32)
                    loss = criterion(outputs['logits'], targets)
                    acc, _ = metric.calculate(outputs['logits'], targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.optimizer != 'sgd':
                    lr_scheduler.step()

                train_loss += loss.item()
                train_acc += acc
                pbar.update(1)
                

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_acc / len(train_loader)

        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)

        print(f"Epoch: {epoch+1} | train loss: {epoch_train_loss:.4f} | train acc: {epoch_train_acc:.4f}% | time: {elapse_time}")
        epoch_valid_loss, epoch_valid_acc, epoch_valid_add = validate(model, criterion, metric, valid_loader, task, device)
        
        results1.append(epoch_valid_acc)
        results2.append(epoch_valid_add)
        train_results1.append(epoch_train_loss)
        train_results2.append(epoch_train_acc)

        print(f"Epoch: {epoch+1} | valid loss: {epoch_valid_loss:.4f} | valid acc: {epoch_valid_acc:.4f}% | metric: {epoch_valid_add:.4f}%")
        
        if epoch_valid_acc > best_acc:
                best_add = epoch_valid_add * 100
                best_acc = epoch_valid_acc
                torch.save({
                    'epoch':epoch,
                    'model_state_dict': model.module.state_dict(),
                    'best_metric':best_add,
                }, p.join(ckpt_path, f'{task}_{args.learning_rate}_{args.seed}.pt'))
        

    results = {
        'train_loss':train_results1,
        'train_acc':train_results2,
        'valid_acc':results1,
        'valid_metric':results2
    }
    with open(f'{task}_{args.learning_rate}_{args.seed}.json', 'w') as f:
        json.dump(results, f)

def train_swa(model, 
          criterion, 
          metric, 
          optimizer, 
          lr_scheduler,
          swa_optim,
          swa_epoch,
          train_loader, 
          valid_loader,
          epochs,
          task,
          device,
          ckpt_path,
          args):
    
    best_acc = 0.0
    best_add = 0.0
    swa_model, swa_scheduler = swa_optim
    train_results1 = []
    train_results2 = []
    valid_results1 = []
    valid_results2 = []
    for epoch in range(epochs):

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
                targets = data['label'].to(torch.int64).cuda(device)

                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

                with torch.autocast('cuda'):
                    if task == 'stsb':
                        outputs['logits'] = outputs['logits'].to(torch.float32)
                        targets = targets / 5.0
                        targets = targets.to(torch.float32)
                    loss = criterion(outputs['logits'], targets)
                    acc, _ = metric.calculate(outputs['logits'], targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += acc
                pbar.update(1)
    
                if epoch <= swa_epoch:      
                    lr_scheduler.step()
        
        if epoch > swa_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step() 

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_acc / len(train_loader)

        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)

        print(f"Epoch: {epoch+1} | train loss: {epoch_train_loss:.4f} | train acc: {epoch_train_acc:.4f}% | time: {elapse_time}")
        
        if epoch > swa_epoch:
            epoch_valid_loss, epoch_valid_acc, epoch_valid_add = validate(swa_model, criterion, metric, valid_loader, task, device)
        else:
            epoch_valid_loss, epoch_valid_acc, epoch_valid_add = validate(model, criterion, metric, valid_loader, task, device)
        
        print(f"Epoch: {epoch+1} | valid loss: {epoch_valid_loss:.4f} | valid acc: {epoch_valid_acc:.4f}% | metric: {epoch_valid_add:.4f}%")

        train_results1.append(epoch_train_loss)
        train_results2.append(epoch_train_acc)
        valid_results1.append(epoch_valid_acc)
        valid_results2.append(epoch_valid_add)

        if epoch_valid_acc > best_acc:
            best_add = epoch_valid_add * 100
            best_acc = epoch_valid_acc
            if epoch > swa_epoch:
                torch.optim.swa_utils.update_bn(train_loader, swa_model)
                torch.save({
                    'epoch':epoch,
                    'model_state_dict': swa_model.state_dict(),
                    'best_acc':best_acc
                    }, p.join(ckpt_path, f'{task}_{args.seed}_swa.pt'))
    
    results = {
        'train_loss':train_results1,
        'train_acc':train_results2,
        'valid_acc':valid_results1,
        'valid_metric':valid_results2
    }
    with open(f'swa_{task}_{args.seed}.json', 'w') as f:
        json.dump(results, f)
    torch.save({
        'epoch':epoch,
        'model_state_dict':swa_model.state_dict(),
        'best_acc': best_acc,
    }, p.join(ckpt_path, f'{task}_{args.seed}_swa.pt'))