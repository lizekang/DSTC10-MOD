from transformers import * 
import os 
import torch 
import json
import numpy as np 

from model import MemeDialoGPT 
from dataset import MODDataset, get_data 
from utils import accuracy_compute, AverageMeter 

import torch.distributed as dist 
from apex import amp 
from apex.parallel import convert_syncbn_model 
from apex.parallel import DistributedDataParallel 
from argparse import ArgumentParser 
from torch.utils.data import DataLoader 
import random 


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[IMG]', '[TAG]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[speaker1]', '[speaker2]', '[IMG]', '[TAG]'], 'pad_token':'[PAD]'}

# data parameters
train_data_path = 'data/dialog/en_data.json'
#train_data_path = 'data/dialog/toy_data.json' 
val_data_path = 'data/dialog/toy_data.json' 
feature_path = 'data/meme/id2feature.json'
#feature_path = 'data/meme/id2feature.json'


# model parameters
use_cuda = torch.cuda.is_available() 
device = torch.device('cuda' if use_cuda else 'cpu') 
model_path = 'ckpt/mod_gpt' 
gpt_path = 'ckpt/origin_gpt'
ckpt_usage = False  
lr = 6e-5
epochs = 8
gradient_accumulation_steps = 5
print_freq = 100 


def main(): 
    
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help='-1 if not distributed') 
    parser.add_argument("--fp16", type=int, default=1, help='O0, O1, O2, or O3') 
    args = parser.parse_args()

    random.seed(0) 
    torch.manual_seed(0) 
    np.random.seed(0) 

    if args.local_rank != -1: 
        dist.init_process_group(backend='nccl', init_method='env://') 
        torch.cuda.set_device(args.local_rank) 
    map_location = "cuda:" + str(args.local_rank) 

    # model initialize 
    if ckpt_usage == True: 
        tokenizer = GPT2Tokenizer.from_pretrained('ckpt/mod_gpt') 
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
        model_config = GPT2Config.from_pretrained('ckpt/mod_gpt') 
        model = MemeDialoGPT(model_config)

    else:
        tokenizer = GPT2Tokenizer.from_pretrained(gpt_path, do_lower_case=True)
        model = MemeDialoGPT.from_pretrained(gpt_path)
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
        model.resize_token_embeddings(len(tokenizer))
    
    if args.fp16: 
        model = convert_syncbn_model(model) 

    
    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=lr)  

    if args.fp16: 
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1') 
    
    if args.local_rank != -1: 
        if args.fp16: 
            model = DistributedDataParallel(model, delay_allreduce=True) 
        else: 
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    if ckpt_usage == True: 
        ckpt_path = 'ckpt/mod_gpt/model.bin' 
        ckpt = torch.load(ckpt_path, map_location=map_location) 
        model.module.load_state_dict(ckpt['model'])
        
    # data read 
    train_dialogs, id2feature = get_data(tokenizer, train_data_path, feature_path) 
    # print(len(train_dialogs))
    val_dialogs, _ = get_data(tokenizer, val_data_path, feature_path) 

    train_dataset = MODDataset(train_dialogs, id2feature, tokenizer) 
    val_dataset = MODDataset(val_dialogs, id2feature, tokenizer) 
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.sampler.SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, sampler=val_sampler) 

    for epoch in range(epochs): 
        
        # one epoch's training
        train_loss = train(args=args, model=model, tokenizer=tokenizer, optimizer=optimizer, dataset=train_loader, epoch=epoch) 
        
        # one epoch's validation 
        validate(model=model, tokenizer=tokenizer, dataset=val_loader, epoch=epoch)
        #break

        # save checkpoint 
        if args.local_rank == 0: 
            torch.save({'model':model.module.state_dict(), 'optimizer': optimizer.state_dict()},\
                '%s/new_epoch_%d_loss_%.3f'%(model_path, epoch, train_loss))
            model.module.config.to_json_file(os.path.join(model_path, 'config.json'))
            tokenizer.save_vocabulary(model_path)


def train(args, model, tokenizer, optimizer, dataset, epoch): 
    model.train() 
    
    avg_loss = AverageMeter() 
    avg_acc = AverageMeter() 
    iteration = 1

    for instance in dataset: 
        history_txt, history_img, token_type_ids, labels = instance 
        history_txt, history_img, token_type_ids, labels = history_txt.to(device).squeeze(0), history_img.to(device).squeeze(0), token_type_ids.to(device).squeeze(0), labels.to(device).squeeze(0)  
        history_txt_embs = model.module.transformer.wte(history_txt) 
        #print(history_txt_embs.size()) 
        history_img_embs = model.module.img_ff(history_img) 
        #print(history_img_embs.size()) 
        #print(token_type_ids) 
        #print(history_txt)
        input_embs = input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer) 
        input_embs = input_embs.to(device) 
        img_feature = history_img[-1, :].unsqueeze(0)
        # print(input_embs.size()) 
        # print(img_feature.size()) 
        loss, lm_logits, _ = model(input_embs, token_type_ids, labels, img_feature) 
        
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward() 
        else: 
            loss.backward()
        

        if iteration % gradient_accumulation_steps == 0:
            if args.fp16: 
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0) 
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            optimizer.zero_grad()
            
        acc = accuracy_compute(lm_logits, labels, 5)
        avg_acc.update(acc)
        avg_loss.update(loss.item())
        
        # print status 
        if iteration % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, iteration, len(dataset),loss=avg_loss, acc=avg_acc))
        
        iteration += 1 

        # print(loss)
        # break 
    return avg_loss.avg  


# concatenate the input 
def input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer): 
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1]) 
    emb_length = token_type_ids.size(-1) 
    emb_dim = history_txt_embs.size(-1)
    img_num = history_img_embs.size(0) 

    input_embs = torch.zeros((emb_length, emb_dim)) 

    txt_idx = 0 
    img_idx = 0 
    left_idx  = 0 
    right_idx = 0 
    while right_idx < emb_length: 
        #if right_idx == emb_length-1 and token_type_ids[right_idx] == img: 
        #    break 
        if right_idx < emb_length-1 and token_type_ids[right_idx] == img:
            txt_length = right_idx - left_idx 
            input_embs[left_idx:right_idx, :] = history_txt_embs[txt_idx:txt_idx+txt_length, :] 
            txt_idx += txt_length 
            input_embs[right_idx,:] = history_img_embs[img_idx, :] 
            img_idx += 1
            left_idx = right_idx + 1 
        right_idx += 1
    txt_length = right_idx - left_idx 
    if txt_length > 0: 
        input_embs[left_idx:right_idx, :] = history_txt_embs[txt_idx:, :]
    # img_feature = history_img_embs[img_idx,:] 
    return input_embs


def validate(model, tokenizer, dataset, epoch): 
    
    model.eval() 
    avg_loss = AverageMeter() 
    avg_acc = AverageMeter() 
    avg_bleu = AverageMeter() 
    iteration = 1
    cat_img_features = img_feature_read(feature_path) 
    meme_correct_num = 0 
    meme_total_num = 0

    with torch.no_grad(): 
        for instance in dataset: 
            history_txt, history_img, token_type_ids, labels = instance 
            history_txt, history_img, token_type_ids, labels = history_txt.to(device).squeeze(0), history_img.to(device).squeeze(0), token_type_ids.to(device).squeeze(0), labels.to(device).squeeze(0)  
            history_txt_embs = model.module.transformer.wte(history_txt) 
            history_img_embs = model.module.img_ff(history_img) 
            
            input_embs = input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer) 
            input_embs = input_embs.to(device) 
            if input_embs.size(-2) > 450:
                continue
            img_feature = history_img[-1, :].unsqueeze(0) 
            loss, lm_logits, cur_img_feature = model(input_embs, token_type_ids, labels, img_feature) 
            # compare cur_img_feature is among topk with img_feature 
            # print(cur_img_feature.size()) 
            if img_feature[0][0] != 0.: 
                if meme_retrieval_compute(cur_img_feature, img_feature, cat_img_features):
                    meme_correct_num += 1 
                meme_total_num += 1 
            acc = accuracy_compute(lm_logits, labels, k=5) 
            avg_acc.update(acc) 
            avg_loss.update(loss.item()) 
            if iteration % print_freq == 0:
                print('Epoch:[{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                'Meme Acc {mac:.3f}'.format(epoch, iteration, len(dataset),loss=avg_loss, acc=avg_acc, mac=float(meme_correct_num/meme_total_num))) 
            iteration += 1 
            #break 


def img_feature_read(feature_path): 
    with open(feature_path, 'r', encoding='utf-8') as f: 
        id2feature_dict = json.load(f) 
    img_features = [] 
    for id in id2feature_dict.keys():
        img_features.append(id2feature_dict[id]) 
    img_features = np.array(img_features) 
    img_features = torch.from_numpy(img_features).float().to(device) 
    return img_features 


def meme_retrieval_compute(cur_img_feature, target_img_feature, cat_img_features): 
    # (1, 512)
    cur_dist = torch.dist(cur_img_feature, target_img_feature, p=2)
    # print(cat_img_features.size())
    cur_img_list = cur_img_feature.repeat(307,1) 
    total_dist = torch.sqrt(torch.sum((cur_img_list - cat_img_features)**2, dim=1))
    # print(total_dist) 
    sorted_total, _ = torch.sort(total_dist) 
    # print(sorted_total) 
    return torch.gt(sorted_total[30],cur_dist)

    # print(cur_dist)



if __name__ == '__main__': 
    main()
