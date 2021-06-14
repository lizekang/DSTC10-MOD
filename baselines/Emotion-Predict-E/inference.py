from transformers import * 
import torch 
import torch.nn.functional as F 
import numpy as np 
from model import MemeDialoGPT 
from dataset import get_data, build_input_from_segments 
import copy 
# from train import input_construct 


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[IMG]', '[TAG]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[speaker1]', '[speaker2]', '[IMG]', '[TAG]'], 'pad_token':'[PAD]'}

# top-k sampling 
def sample_sequence(input_embs, token_type_ids, model, tokenizer, speaker_id, max_len=20): 
    temperature = 0.7 
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1]) 
    res = [] 
    for i in range(max_len): 
        logits, _ = model(input_embs, token_type_ids) 
        logits = logits[-1]/temperature  
        # print(logits.size()) 
        logits = top_filtering(logits, top_k=0, top_p=0.9) 
        probs = F.softmax(logits, dim=-1) 
        next_word = torch.multinomial(probs, 1).item() 
        if next_word == eos or next_word == 2: 
            break 
        res.append(next_word) 
        token_type_ids = torch.cat((token_type_ids, torch.LongTensor([speaker_id])), 0) 
        word_emb = model.transformer.wte(torch.LongTensor([next_word])) 
        input_embs = torch.cat((input_embs, word_emb), 0) 
        #break 
    
    return res 



# select top-k or top-p candidates
def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')): 
    assert logits.dim()==1 
    top_k = min(top_k, logits.size(-1)) 
    if top_k > 0:
        idxs_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[idxs_to_remove] = filter_value 
    if top_p > 0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True) 
        cummulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) 

        sorted_idx_to_remove = cummulative_probs > top_p 
        sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[...,:-1].clone()
        sorted_idx_to_remove[...,0] = 0 

        idxs_to_remove = sorted_idx[sorted_idx_to_remove]
        logits[idxs_to_remove] = filter_value 
    idxs_to_remove = logits < threshold 
    logits[idxs_to_remove] = filter_value
    # print(logits.size())
    return logits




def generate_response(model, dialog_list, id2feature, tokenizer): 
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1]) 
    with torch.no_grad(): 
        for dialog in dialog_list: 
            history = copy.deepcopy(dialog['history']) 
            history_txt, history_img, token_type_ids, _ = build_input_from_segments(history, tokenizer, id2feature) 
            if token_type_ids[-1] == speaker1:
                speaker_id = speaker2
            else:
                speaker_id = speaker1 
            
            history_txt += [speaker_id] 
            token_type_ids += [speaker_id]

            if len(history_img)==0:
                continue  
            print(tokenizer.convert_ids_to_tokens(history_txt))

            history_txt = torch.LongTensor(history_txt) 
            history_img = torch.from_numpy(np.array(history_img)).float() 
            token_type_ids = torch.Tensor(token_type_ids).long()
            # print(token_type_ids.size(), history_txt.size(), history_img.size())

            history_txt_embs = model.transformer.wte(history_txt) 
            history_img_embs = model.img_ff(history_img) 

            input_embs = input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer) 
            # print(input_embs.size())
            res = sample_sequence(input_embs, token_type_ids, model, tokenizer, speaker_id) 
            print(tokenizer.convert_ids_to_tokens(res))
            break 


            


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
        if token_type_ids[right_idx] == img:
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



if __name__ == '__main__': 
    ckpt_path = 'ckpt/mod_gpt'
    tokenizer = BertTokenizer.from_pretrained(ckpt_path, do_lower_case=True) 
    model_config = GPT2Config.from_pretrained(ckpt_path) 
    model = MemeDialoGPT(model_config) 
    ckpt = torch.load('ckpt/mod_gpt/model.bin', map_location='cpu') 
    model.load_state_dict(ckpt['model'])  
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 

    use_cuda = torch.cuda.is_available() 
    device = torch.device('cuda' if use_cuda else 'cpu') 
    model = model.to(device) 
    model.eval() 

    test_path = 'data/dialog/validation.json' 
    feature_path = 'data/meme/id2feature.json' 
    #test_data = json.load(open(test_path, 'r', encoding='utf-8')) 
    dialog_list, id2feature = get_data(tokenizer, test_path, feature_path) 
    # print(dialog_list[0]) 
    generate_response(model, dialog_list, id2feature, tokenizer) 
