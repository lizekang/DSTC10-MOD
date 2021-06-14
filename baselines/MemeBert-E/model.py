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
        # predict the meme usage 
        self.lm_flag = nn.Linear(config.n_embd, 2)

    def tie_weights(self): 
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte) 
    
    def forward(self, input_embeds, token_type_ids, labels=None, img_feature=None, meme_flag=None): 
        transformer_outputs = self.transformer(inputs_embeds=input_embeds, token_type_ids=token_type_ids) 
        hidden_states = transformer_outputs[0] 
        txt_hidden_states, img_hidden_states = hidden_states[:-1, :], hidden_states[-1, :].unsqueeze(0) 

        lm_logits = self.lm_head(txt_hidden_states) 
        img_regs = self.img_inverse_ff(img_hidden_states) 
        outputs = (lm_logits,) + (img_regs, ) 


        if labels is not None: 
            txt_loss_fct = CrossEntropyLoss(ignore_index=-100) 
            loss = txt_loss_fct(lm_logits, labels)  

            if meme_flag is not None:
                mf_logits = self.lm_flag(img_hidden_states) 
                mf_loss_fct = CrossEntropyLoss()
                mf_flag_loss = mf_loss_fct(mf_logits, meme_flag) 
                loss += mf_flag_loss 
                outputs = (mf_logits,) + outputs

            if img_feature[0][0] != 0.:  
                img_loss_fct = MSELoss() 
                img_loss = img_loss_fct(img_regs, img_feature) 
                print(img_loss)
                loss += img_loss
            outputs = (loss,) + outputs 
        return outputs   



class MemeBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 274
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) 
        # print(logits.size())

        loss = None
        if labels is not None: 
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return (loss,) + (logits,)


