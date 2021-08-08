from numpy.lib.function_base import disp
from transformers import *
import os
import torch
import json
import numpy as np

from parallel_model import MemeDialoGPT
from dataset import MODDataset, get_data
from utils import accuracy_compute, AverageMeter, meme_classify_accuracy

import torch.distributed as dist
# from apex import amp
# from apex.parallel import convert_syncbn_model
# from apex.parallel import DistributedDataParallel
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import random
from utils import get_logger, try_create_dir
import logging

SPECIAL_TOKENS = [
    '[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[IMG]', '[TAG]', '[PAD]'
]
SPECIAL_TOKENS_DICT = {
    'bos_token': '[BOS]',
    'eos_token': '[EOS]',
    'additional_special_tokens':
    ['[speaker1]', '[speaker2]', '[IMG]', '[TAG]'],
    'pad_token': '[PAD]'
}

data_dir = '../../data'

train_data_path = os.path.join(data_dir, 'dialog/train.json')
# train_data_path = 'debug.json'
val_data_path = os.path.join(data_dir, 'dialog/validation.json')
# val_data_path = 'debug.json'
feature_path = os.path.join(data_dir, 'meme/id2feature.json')

# model parameters
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
model_path = 'ckpt/mod_gpt'
# gpt_path = 'ckpt/origin_gpt'
try_create_dir(model_path)
gpt_path = 'ckpt/gpt2-chinese-cluecorpussmall'

ckpt_usage = False
ckpt_path = './ckpt/mod_gpt/epoch_0_loss_10.701'
start_epoch = 0

lr = 6e-5
epochs = 35
gradient_accumulation_steps = 8
print_freq = 100
logger = get_logger(__name__)
logger.info(f"device:{device}")
logger.debug(f"torch version:{torch.__version__}")


def main():

    parser = ArgumentParser()
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help='-1 if not distributed')
    parser.add_argument("--fp16",
                        type=int,
                        default=0,
                        help='O0, O1, O2, or O3')
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
        tokenizer = BertTokenizer.from_pretrained('ckpt/mod_gpt',
                                                  do_lower_case=True)
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        model_config = GPT2Config.from_pretrained('ckpt/mod_gpt')
        model = MemeDialoGPT(model_config)
        # important!!! influence the length of model.named_parameters() and thus influence optimizer loading
        model.tie_weights()

    else:
        tokenizer = BertTokenizer.from_pretrained(gpt_path, do_lower_case=True)
        logger.info(f"vocab len:{len(tokenizer)}")
        model = MemeDialoGPT.from_pretrained(gpt_path)
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"vocab len:{len(tokenizer)}")
    if args.fp16:
        model = convert_syncbn_model(model)

    model = model.to(device)
    # model.eval()

    logger.debug('after creating optimizer')

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if args.fp16:
        model = DistributedDataParallel(model, delay_allreduce=True)
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)
        # find_unused_parameters=False)

    optimizer = AdamW(model.parameters(), lr=lr)
    if ckpt_usage:
        ckpt = torch.load(ckpt_path, map_location=map_location)
        model.module.load_state_dict(ckpt['model'])
        # for name, v in model.named_parameters():
        #     print(f"{name}, {v.size()}")
        # print(len(list(model.named_parameters())))
        optimizer.load_state_dict(ckpt['optimizer'])

    logger.debug('after creating parallel model')

    def display(d, level=0):
        for k, v in d.items():
            if isinstance(v, dict):
                print('*'*level+f'{k}:')

                display(v, level+1)
            elif isinstance(v, torch.Tensor):
                print('*'*level+f'{k}:{v.size()}')

            else:
                print('*'*level+f'{k}:{v}')

    # if ckpt_usage:
    #     ckpt = torch.load(ckpt_path, map_location=map_location)
    #     # logger.debug(model.module)
    #     model.module.load_state_dict(ckpt['model'])
    #     display(optimizer.state_dict())
    #     display(ckpt['optimizer'])
    #     # exit()
    #     # for k, v in optimizer.state_dict().items():
    #     #     logger.debug(f"{k}:{v}")
    #     # logger.debug('===state_dict===')
    #     # for k, v in ckpt['optimizer'].items():
    #     #     logger.debug(f"{k}:{v}")
    #     # optimizer.load_state_dict(ckpt['optimizer'])
    #     logger.info('ckpt_usage True, load model and optimizer succ, start epoch:', start_epoch)

    # if ckpt_usage == True:
    #     ckpt_path = 'ckpt/mod_gpt/model.bin'
    #     ckpt = torch.load(ckpt_path, map_location=map_location)
    #     model.module.load_state_dict(ckpt['model'])

    # data read
    logger.debug('before get_data')
    train_dialogs, id2feature = get_data(tokenizer, train_data_path,
                                         feature_path)
    # print(len(train_dialogs))
    val_dialogs, _ = get_data(tokenizer, val_data_path, feature_path)
    logger.debug('after get_data')

    train_dataset = MODDataset(train_dialogs, id2feature, tokenizer)
    val_dataset = MODDataset(val_dialogs, id2feature, tokenizer)
    logger.debug('after dataset')
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    val_sampler = torch.utils.data.sampler.SequentialSampler(val_dataset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              num_workers=8,
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=8,
                            sampler=val_sampler)

    logger.info('finish load data')
    # for name, v in model.named_parameters():
    #     print(f"{name}, {v.size()}")
    # print(len(list(model.named_parameters())))
    for epoch in range(start_epoch, epochs):

        # one epoch's training
        train_sampler.set_epoch(epoch)
        train_loss = train(args=args,
                           model=model,
                           tokenizer=tokenizer,
                           optimizer=optimizer,
                           dataset=train_loader,
                           epoch=epoch)

        # one epoch's validation
        validate(model=model,
                 tokenizer=tokenizer,
                 dataset=val_loader,
                 epoch=epoch)
        # break
        # torch.distributed.barrier()

        # save checkpoint
        logger.info(f"epoch:{epoch}, local rank: {args.local_rank}")

        if args.local_rank == 0:
            # for name, v in model.named_parameters():
            #     print(f"{name}, {v.size()}")
            # print(len(list(model.named_parameters())))
            # print(len(list(model.module.named_parameters())))

            logger.info(f"epoch:{epoch}, begin to save")
            torch.save({'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()},
                       '%s/epoch_%d_loss_%.3f' % (model_path, epoch, train_loss))
            model.module.config.to_json_file(
                os.path.join(model_path, 'config.json'))
            tokenizer.save_vocabulary(model_path)
            logger.info(f"epoch:{epoch}, finish save")

        # torch.distributed.barrier()


def train(args, model, tokenizer, optimizer, dataset, epoch):
    model.train()

    avg_loss = AverageMeter()
    avg_img_loss = AverageMeter()
    avg_text_loss = AverageMeter()
    avg_acc_5 = AverageMeter()
    avg_acc_30 = AverageMeter()
    avg_acc_90 = AverageMeter()
    iteration = 0
    # cat_img_features = img_feature_read(feature_path)
    meme_correct_num = 1
    meme_total_num = 1

    for instance in dataset:
        history_txt, history_img, token_type_ids, labels, meme_flag, id_labels = instance
        history_txt, history_img, token_type_ids, labels, meme_flag, id_labels = history_txt.to(device).squeeze(0), history_img.to(device).squeeze(0), \
            token_type_ids.to(device).squeeze(0), labels.to(device).squeeze(
                0), meme_flag.to(device).squeeze(0), id_labels.to(device).squeeze(0)
        history_txt_embs = model.module.transformer.wte(history_txt)
        # print(history_txt_embs.size())
        history_img_embs = model.module.img_ff(history_img)
        # print(history_img_embs.size())
        # print(token_type_ids)
        # print(history_txt)
        input_embs = input_construct(history_txt_embs, history_img_embs,
                                     token_type_ids, tokenizer)
        input_embs = input_embs.to(device)
        if input_embs.size(-2) > 450:
            input_embs = input_embs[-450:, :]
            token_type_ids = token_type_ids[-450:]
            labels = token_type_ids[-449:]
            # continue
        img_feature = history_img[-1, :].unsqueeze(0)
        # logger.debug(f"{input_embs.size()}, {token_type_ids.size()}, {labels.size()}, {img_feature.size()}, {meme_flag.size()}")

        # print(input_embs.size())
        # print(img_feature.size())
        loss, img_loss, text_loss = model(input_embs, token_type_ids, id_labels, labels, img_feature,
                                                  meme_flag)
        logits = model.module.logits
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()

        if iteration % gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                               1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # if img_feature[0][0] != 0.:
        #    if meme_retrieval_compute(cur_img_feature, img_feature, cat_img_features):
        #        meme_correct_num += 1
        #    meme_total_num += 1
        #acc = accuracy_compute(lm_logits, labels, 5)
        # avg_acc.update(acc)
        if id_labels.numel() > 0:
            acc_5, acc_30, acc_90 = acc_compute(logits, id_labels)
            avg_acc_5.update(acc_5)
            avg_acc_30.update(acc_30)
            avg_acc_90.update(acc_90)
        avg_loss.update(loss.item())
        if img_loss.item() > 0:
            assert id_labels.numel() > 0
            avg_img_loss.update(img_loss.item())
        avg_text_loss.update(text_loss.item())

        # print status
        if iteration % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Loss {loss.avg:.4f} Image Loss {img_loss.avg:.4f} Text Loss {text_loss.avg:.4f}\t'
                  'Retrieval Acc {acc_5.avg:.3f} | {acc_30.avg:.3f} | {acc_90.avg:.3f}'.format(epoch, iteration, len(dataset),
                                                                                               loss=avg_loss, img_loss=avg_img_loss, text_loss=avg_text_loss, acc_5=avg_acc_5, acc_30=avg_acc_30, acc_90=avg_acc_90))

        iteration += 1
        # logger.info(f"iteration:{iteration}, local rank: {args.local_rank}")

        # print(loss)
        # break
    return avg_loss.avg


def acc_compute(logits, labels):
    _, idx = torch.sort(logits.squeeze(0))
    idx = idx.tolist()
    labels = labels.item()
    return int(labels in idx[-5:]), int(labels in idx[-30:]), int(labels in idx[-90:])


# concatenate the input
def input_construct(history_txt_embs, history_img_embs, token_type_ids,
                    tokenizer):
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-1])
    emb_length = token_type_ids.size(-1)
    emb_dim = history_txt_embs.size(-1)
    img_num = history_img_embs.size(0)

    input_embs = torch.zeros((emb_length, emb_dim))

    txt_idx = 0
    img_idx = 0
    left_idx = 0
    right_idx = 0
    while right_idx < emb_length:
        # if right_idx == emb_length-1 and token_type_ids[right_idx] == img:
        #    break
        if right_idx < emb_length - 1 and token_type_ids[right_idx] == img:
            txt_length = right_idx - left_idx
            input_embs[left_idx:right_idx, :] = history_txt_embs[
                txt_idx:txt_idx + txt_length, :]
            txt_idx += txt_length
            input_embs[right_idx, :] = history_img_embs[img_idx, :]
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
    avg_img_loss = AverageMeter()
    avg_text_loss = AverageMeter()
    avg_acc_5 = AverageMeter()
    avg_acc_30 = AverageMeter()
    avg_acc_90 = AverageMeter()
    avg_bleu = AverageMeter()
    iteration = 1
    cat_img_features = img_feature_read(feature_path)
    meme_correct_num = 0
    meme_total_num = 0

    with torch.no_grad():
        for instance in dataset:
            history_txt, history_img, token_type_ids, labels, meme_flag, id_labels = instance
            history_txt, history_img, token_type_ids, labels, meme_flag, id_labels = history_txt.to(device).squeeze(0), history_img.to(device).squeeze(0), \
                token_type_ids.to(device).squeeze(0), labels.to(device).squeeze(
                    0), meme_flag.to(device).squeeze(0), id_labels.to(device).squeeze(0)
            history_txt_embs = model.module.transformer.wte(history_txt)
            history_img_embs = model.module.img_ff(history_img)

            input_embs = input_construct(history_txt_embs, history_img_embs,
                                         token_type_ids, tokenizer)
            input_embs = input_embs.to(device)
            if input_embs.size(-2) > 450:
                continue
            img_feature = history_img[-1, :].unsqueeze(0)
            loss, img_loss, text_loss = model(input_embs, token_type_ids, id_labels, labels, img_feature,
                                                      meme_flag)
            logits = model.module.logits
            if id_labels.numel() > 0:
                acc_5, acc_30, acc_90 = acc_compute(logits, id_labels)
                avg_acc_5.update(acc_5)
                avg_acc_30.update(acc_30)
                avg_acc_90.update(acc_90)
            avg_loss.update(loss.item())
            if img_loss.item() > 0:
                avg_img_loss.update(img_loss.item())
            avg_text_loss.update(text_loss.item())

            # print status
            if iteration % print_freq == 0:
                print('Epoch:[{0}][{1}/{2}]\t'
                      'Loss {loss.avg:.4f} Image Loss {img_loss.avg:.4f} Text Loss {text_loss.avg:.4f}\t'
                      'Retrieval Acc {acc_5.avg:.3f} | {acc_30.avg:.3f} | {acc_90.avg:.3f}'.format(epoch, iteration, len(dataset),
                                                                                                   loss=avg_loss, img_loss=avg_img_loss, text_loss=avg_text_loss, acc_5=avg_acc_5, acc_30=avg_acc_30, acc_90=avg_acc_90))

            iteration += 1

            # loss, mf_logits, lm_logits, cur_img_feature = model(
            #     input_embs, token_type_ids, labels, img_feature, meme_flag,
            #     'val')
            # # compare cur_img_feature is among topk with img_feature
            # # print(cur_img_feature.size())
            # if img_feature[0][0] != 0.:
            #     if meme_retrieval_compute(cur_img_feature, img_feature,
            #                               cat_img_features):
            #         meme_correct_num += 1
            #     meme_total_num += 1
            # #acc = accuracy_compute(lm_logits, labels, k=5)
            # acc = meme_classify_accuracy(mf_logits, meme_flag).item()
            # avg_acc.update(acc)
            # avg_loss.update(loss.item())
            # if iteration % print_freq == 0:
            #     print('Epoch:[{0}][{1}/{2}]\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Meme Classification {acc.val:.3f} ({acc.avg:.3f})\t'
            #           'Meme Retrieval {mac:.3f}'.format(
            #               epoch,
            #               iteration,
            #               len(dataset),
            #               loss=avg_loss,
            #               acc=avg_acc,
            #               mac=float(meme_correct_num / meme_total_num)))
            # iteration += 1
            # break

    logger.info(
        f"validate epoch {epoch} end, Loss {avg_loss.avg}, Meme Retrieval {avg_acc_5.avg} | {avg_acc_30.avg} | {avg_acc_90.avg}"
    )


def img_feature_read(feature_path):
    with open(feature_path, 'r', encoding='utf-8') as f:
        id2feature_dict = json.load(f)
    img_features = []
    for id in id2feature_dict.keys():
        img_features.append(id2feature_dict[id])
    img_features = np.array(img_features)
    img_features = torch.from_numpy(img_features).float().to(device)
    return img_features


def meme_retrieval_compute(cur_img_feature, target_img_feature,
                           cat_img_features):
    # (1, 512)
    cur_dist = torch.dist(cur_img_feature, target_img_feature, p=2)
    # print(cat_img_features.size())
    cur_img_list = cur_img_feature.repeat(307, 1)
    total_dist = torch.sqrt(
        torch.sum((cur_img_list - cat_img_features)**2, dim=1))
    # print(total_dist)
    sorted_total, _ = torch.sort(total_dist)
    # print(sorted_total)
    return torch.gt(sorted_total[30], cur_dist)

    # print(cur_dist)


if __name__ == '__main__':
    main()
