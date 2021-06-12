import json 
from transformers import * 
import torch 
from torch.utils.data import Dataset 
import numpy as np 
import copy 



SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[IMG]', '[TAG]', '[CLS]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[speaker1]', '[speaker2]', '[IMG]', '[TAG]'], 'pad_token':'[PAD]'}


def tokenize(obj, tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n,o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj) 
    

# split the dataset into form of histroy and answer 
def get_data(tokenizer, data_path, meme_feature_path):
    dialog_data = json.load(open(data_path, 'r', encoding='utf-8')) 
    dialog_list = [] 
    for idx in dialog_data.keys():
        dialog = dialog_data[idx] 
        history = [] 
        for i in range(len(dialog)): 
            if 'txt' in dialog[i].keys(): 
                dialog[i]['txt'] = tokenize(dialog[i]['txt'], tokenizer) 
            if i == 0: 
                history.append(dialog[i]) 
                continue 
            pair = {'history': copy.deepcopy(history), 'answer': copy.deepcopy(dialog[i])} 
            dialog_list.append(pair) 
            history.append(dialog[i]) 
        # break 
    id2feature = json.load(open(meme_feature_path, 'r', encoding='utf-8')) 
    return dialog_list, id2feature 


def get_emotion_data(tokenizer, data_path, meme_feature_path): 
    dialog_data = json.load(open(data_path, 'r', encoding='utf-8')) 
    dialog_list = [] 
    for idx in dialog_data.keys():
        dialog = dialog_data[idx] 
        history = [] 
        for i in range(len(dialog)):
            if 'txt' in dialog[i].keys():
                dialog[i]['txt'] = tokenize(dialog[i]['txt'], tokenizer) 
            history.append(dialog[i])
            if 'emotion_id' in dialog[i].keys():
                pair = {'situation': copy.deepcopy(history), 'emotion': copy.deepcopy(dialog[i]['emotion_id'])}
                dialog_list.append(pair)
    id2feature = json.load(open(meme_feature_path, 'r', encoding='utf-8')) 
    return dialog_list, id2feature 



class MODDataset(Dataset): 
    def __init__(self, dialogs, id2feature, tokenizer): 
        self.dialogs = dialogs 
        self.id2feature = id2feature 
        self.tokenizer = tokenizer 
    
    def __len__(self):
        return len(self.dialogs) 
    
    def __getitem__(self, index): 
        his = copy.deepcopy(self.dialogs[index]['history']) 
        ans = copy.deepcopy(self.dialogs[index]['answer']) 
        #print(his)
        #print(ans)
        history_txt, histroy_img, token_type_ids, labels, meme_flag = build_input_from_segments(history=his, answer=ans, tokenizer=self.tokenizer, id2feature=self.id2feature) 
        history_txt = torch.LongTensor(history_txt) 
        histroy_img = torch.from_numpy(np.array(histroy_img)).float() 
        token_type_ids = torch.Tensor(token_type_ids).long()
        labels = torch.Tensor(labels).long() 
        meme_flag = torch.Tensor(meme_flag).long()

        return history_txt, histroy_img, token_type_ids, labels, meme_flag  


class EmotionDataset(Dataset):
    def __init__(self, dialogs, id2feature, tokenizer): 
        self.dialogs = dialogs 
        self.id2feature = id2feature 
        self.tokenizer = tokenizer 
    
    def __len__(self):
        return len(self.dialogs) 
    
    def __getitem__(self, index): 
        his = copy.deepcopy(self.dialogs[index]['situation']) 
        emo = copy.deepcopy(self.dialogs[index]['emotion']) 
        #print(his)
        #print(ans)
        history_txt, histroy_img, token_type_ids, _, _ = build_input_from_segments(history=his, tokenizer=self.tokenizer, id2feature=self.id2feature) 
        history_txt = torch.LongTensor(history_txt) 
        histroy_img = torch.from_numpy(np.array(histroy_img)).float() 
        token_type_ids = torch.Tensor(token_type_ids).long()
        #labels = torch.Tensor(labels).long() 
        #meme_flag = torch.Tensor(meme_flag).long()
        emo_id = torch.Tensor([emo]).long()

        return history_txt, histroy_img, token_type_ids, emo_id



# build input type from data 
def build_input_from_segments(history, tokenizer, id2feature, answer=None): 
    bos, eos, speaker1, speaker2, img, tag, cls_token = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1]) 
    history_txt = []
    history_img = [] 
    labels = []
    token_type_ids = [] 
    
    # to avoid out of length, cut the partial sequence
    ans_len = 4 
    if answer is not None and 'txt' in answer.keys():
        ans_len += len(answer['txt']) 
    
    for i in range(len(history)-1, -1, -1): 
        
        # evaluate the length 
        cur_len = 4 
        if 'txt' in history[i].keys():
            cur_len += len(history[i]['txt']) 
        if len(token_type_ids) + ans_len + cur_len > 500: 
            break 

        if history[i]['speaker_id'] == '[speaker1]': 
            speaker_id = speaker1 
        else:
            speaker_id = speaker2 
        if 'img_id' in history[i].keys(): 
            history_img = [id2feature[history[i]['img_id']]] + history_img
            token_type_ids = [img] + token_type_ids 
            labels = [-100] + labels 

        if 'txt' in history[i].keys(): 
            content = [bos] + history[i]['txt'] + [eos] 
            history_txt = content + history_txt 
            token_type_ids = [speaker_id] * len(content) + token_type_ids 
            labels = [-100] * len(content) + labels 
        else: 
            content = [bos] + [eos] 
            history_txt = content + history_txt 
            token_type_ids = [speaker_id] * len(content) + token_type_ids 
            labels = [-100] * len(content) + labels 
    
        history_txt = [speaker_id] + history_txt 
        token_type_ids = [speaker_id] + token_type_ids 
        labels = [-100] + labels 
    meme_flag = []

    history_txt += [cls_token] 
    token_type_ids += [cls_token] 
    labels += [-100]
    if answer is not None: 
        if answer['speaker_id'] == '[speaker1]': 
            speaker_id = speaker1 
        else:
            speaker_id = speaker2 
    
        history_txt += [speaker_id] 
        token_type_ids += [speaker_id] 
        if 'txt' in answer.keys(): 
            content = [bos] + answer['txt'] + [eos] 
            history_txt += content 
            token_type_ids += [speaker_id] * len(content) 
            labels += content 
        else: 
            content = [bos] + [eos] 
            history_txt += content 
            token_type_ids += [speaker_id] * len(content) 
            labels += content 
    
        labels += [-100, -100] 
        history_txt += [tag] 
        token_type_ids += [img] 
        if 'img_id' in answer.keys(): 
            history_img += [id2feature[answer['img_id']]] 
            meme_flag = [1]
        else:
            history_img += [[0.0]*512]
            meme_flag = [0] 
    return history_txt, history_img, token_type_ids, labels[1:], meme_flag 


if __name__ == '__main__': 
    data_path = 'data/dialog/validation.json' 
    meme_feature_path = 'data/meme/id2feature.json'
    tokenizer = BertTokenizer.from_pretrained('ckpt/origin_gpt', do_lower_case=True)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    dialog_list, id2feature = get_emotion_data(tokenizer, data_path, meme_feature_path) 
    print(dialog_list[1]) 
    # print(len(id2feature['001'])) 
    #dataset = EmotionDataset(dialog_list, id2feature, tokenizer) 
    #print(dataset[0])
    
    dataset = EmotionDataset(dialog_list, id2feature, tokenizer) 
    history_txt, history_img, token_type_ids, emo_id= dataset[0]
    print(tokenizer.convert_ids_to_tokens(history_txt))
    print(history_img.size())
    print(tokenizer.convert_ids_to_tokens(token_type_ids))
    print(emo_id)
    #rint(meme_flag.size())
    
