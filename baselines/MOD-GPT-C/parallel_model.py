import torch 
from torch import nn 
from torch.nn import CrossEntropyLoss, MSELoss 
import os 
from transformers import * 

class MemeDialoGPT(GPT2PreTrainedModel): 
    def __init__(self, config): 
        super(MemeDialoGPT, self).__init__(config) 
        self.transformer = GPT2Model(config) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
        self.img_ff = nn.Linear(512, config.n_embd) 
        self.img_inverse_ff = nn.Linear(config.n_embd, 512) 
        self.lm_flag = nn.Linear(config.n_embd, 2)

        self.num_labels = 304 
        self.dropout = nn.Dropout(0.1) 
        self.classifier = nn.Linear(config.n_embd, self.num_labels)

    def tie_weights(self): 
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte) 
    
    def forward(self, input_embeds, token_type_ids, id_labels=None, labels=None, img_feature=None, meme_flag=None, mod='train'): 
        transformer_outputs = self.transformer(inputs_embeds=input_embeds, token_type_ids=token_type_ids) 
        hidden_states = transformer_outputs[0] 
        txt_hidden_states, img_hidden_states = hidden_states[:-1, :], hidden_states[-1, :].unsqueeze(0) 

        lm_logits = self.lm_head(txt_hidden_states) 
        img_regs = self.img_inverse_ff(img_hidden_states) 
        outputs = (lm_logits,) + (img_regs,) 

        if id_labels is not None:
            pooled_output = self.dropout(img_hidden_states)
            logits = self.classifier(pooled_output)

            #add txt loss
            txt_loss_fct = CrossEntropyLoss(ignore_index=-100) 
            text_loss = txt_loss_fct(lm_logits, labels)
            if id_labels.numel() > 0:
                loss_fct = CrossEntropyLoss()
                img_loss = loss_fct(logits.view(-1, self.num_labels), id_labels.view(-1)) 
                total_loss = img_loss + text_loss
            else:
                img_loss = torch.zeros_like(text_loss)
                total_loss = img_loss + text_loss
            self.logits = logits
            return (total_loss, img_loss, text_loss)
        
        if labels is not None: 
            raise Exception('duplicated branch!')
            txt_loss_fct = CrossEntropyLoss(ignore_index=-100) 
            loss = txt_loss_fct(lm_logits, labels)  
            #print(loss) 

            if meme_flag is not None:
                mf_logits = self.lm_flag(img_hidden_states) 
                mf_loss_fct = CrossEntropyLoss()
                mf_flag_loss = mf_loss_fct(mf_logits, meme_flag) 
                loss += mf_flag_loss 
                outputs = (mf_logits,) + outputs

            if img_feature[0][0] != 0.:  
                img_loss_fct = MSELoss() 
                loss += img_loss_fct(img_regs, img_feature)
                #print(loss) 
            outputs = (loss,) + outputs 
            if mod == 'train':
                return loss 
        return outputs   
