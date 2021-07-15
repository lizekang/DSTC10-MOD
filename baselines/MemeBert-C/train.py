from transformers import * 
import os 
import torch 
import json
import numpy as np  

from model import MemeBERT 
from dataset import MemeDataset, get_data 
from torch.utils.data import DataLoader  

bert_path = 'ckpt/bert'
train_data_path = 'data/dialog/train.json'
lr = 6e-5 
epochs = 8
use_cuda = torch.cuda.is_available() 
device = torch.device('cuda' if use_cuda else 'cpu') 
gradient_accumulation_steps = 5 
print_freq = 1

def main(): 
    tokenizer = AutoTokenizer.from_pretrained(bert_path) 
    model = MemeBERT.from_pretrained(bert_path) 
    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=lr) 
    
    train_dialogs = get_data(tokenizer, train_data_path) 
    train_dataset = MemeDataset(train_dialogs, tokenizer) 

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8, pin_memory=True) 

    for epoch in range(epochs):
        train(model=model, tokenizer=tokenizer, optimizer=optimizer, dataset=train_loader, epoch=epoch) 
        break 


def train(model, tokenizer, optimizer, dataset, epoch): 
    model.train() 
    avg_loss = AverageMeter() 
    avg_acc = AverageMeter() 
    iteration = 1 

    for instance in dataset: 
        
        history_txt, labels = instance 
        history_txt, labels = history_txt, labels.squeeze(0)
        # print(history_txt.size(), labels.size()) 
        loss, logits  = model(input_ids=history_txt, labels=labels) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        
        if iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


        avg_loss.update(loss.item()) 

        # print(logits.size())
        if acc_compute(logits, labels): 
            acc = 1 
        else:
            acc = 0
        avg_acc.update(acc)
        
        if iteration % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Classify Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, iteration, len(dataset),loss=avg_loss, acc=avg_acc)) 
        iteration += 1 
        break 

def acc_compute(logits, labels):
    _, idx = torch.sort(logits.squeeze(0)) 
    idx = idx.tolist() 
    labels = labels.item()
    return labels in idx[-90:]

# class for evaluation metric 
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__': 
    main()