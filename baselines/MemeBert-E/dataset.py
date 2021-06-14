import json 
from transformers import * 
import torch 
from torch.utils.data import Dataset 
import numpy as np 
import copy  

def tokenize(obj, tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n,o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)  


# split the dataset into form of histroy and answer 
def get_data(tokenizer, data_path):
    dialog_data = json.load(open(data_path, 'r', encoding='utf-8')) 
    dialog_list = [] 
    for idx in dialog_data.keys():
        dialog = dialog_data[idx] 
        history = [] 
        for i in range(len(dialog)): 
            if 'txt' not in dialog[i].keys(): 
                continue 
            dialog[i]['txt'] = tokenize(dialog[i]['txt'], tokenizer) 
            if i == 0: 
                history.append(dialog[i]) 
                continue 
            if 'img_id' not in dialog[i].keys(): 
                history.append(dialog[i]) 
                continue 
            pair = {'history': copy.deepcopy(history), 'answer': copy.deepcopy(dialog[i])} 
            dialog_list.append(pair) 
            history.append(dialog[i]) 
        # break 
    return dialog_list


class MemeDataset(Dataset): 
    def __init__(self, dialogs, tokenizer): 
        self.dialogs = dialogs 
        self.tokenizer = tokenizer 
    
    def __len__(self):
        return len(self.dialogs) 
    
    def __getitem__(self, index): 
        his = copy.deepcopy(self.dialogs[index]['history']) 
        ans = copy.deepcopy(self.dialogs[index]['answer']) 
        #print(his)
        #print(ans)
        history_txt, labels = build_input_from_segments(his, self.tokenizer, ans) 
        try:
            history_txt = torch.LongTensor(history_txt) 
        except:
            print(his)
            print(history_txt)
        labels = torch.Tensor(labels).long()
        

        return history_txt, labels


def build_input_from_segments(history, tokenizer, answer=None):
    cls_token, sep_token = tokenizer.convert_tokens_to_ids(['[CLS]','[SEP]']) 
    history_txt = [] 
    labels = []
    for i in range(len(history)): 
        history_txt += history[i]['txt'] 
    if answer is not None:
        if 'txt' in answer.keys():
            history_txt += answer['txt'] 
        labels.append(int(answer['img_id']))
    if len(history_txt) >= 505: 
        history_txt = history_txt[-500:] 
    history_txt = [cls_token] + history_txt + [sep_token]

    return history_txt, labels


if __name__ == '__main__': 
    data_path = 'data/dialog/validation.json' 
    bert_path = 'ckpt/bert'
    tokenizer = AutoTokenizer.from_pretrained(bert_path) 
    x = 'hellow word'
    print(tokenizer.tokenize(x))
    '''
    dialog_list = get_data(tokenizer, data_path) 
    print(dialog_list[0])
    dataset = MemeDataset(dialog_list, tokenizer) 
    his, lab = dataset[0]
    print(tokenizer.convert_ids_to_tokens(his))
    '''