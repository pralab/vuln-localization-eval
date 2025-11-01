# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch

from models.base_explainable_model import ExplainableVulnerabilityModel

class CodexglueModel(nn.Module, ExplainableVulnerabilityModel):   
    def __init__(self, encoder,config,tokenizer,args):
        super(CodexglueModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
 
    def forward(self, input_ids=None, inputs_embeds=None, labels=None, output_attentions=False, attention_mask=None): 
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id) if attention_mask is None else attention_mask

        if input_ids is not None:
            outputs = self.encoder(input_ids,attention_mask=attention_mask, output_attentions=output_attentions)
        else:
            outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=output_attentions)
        
        logits = outputs.logits

        # Apply dropout
        logits = self.dropout(logits)

        prob = torch.sigmoid(logits)

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return (loss, prob, outputs.attentions) if output_attentions else (loss, prob)
        
        return (prob, outputs.attentions) if output_attentions else prob

    def get_embedding_layer(self) -> nn.Module:
        return self.encoder.roberta.embeddings
    
    def get_reference_input_ids(self, input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
        seq_length = input_ids.size(1)
        ref_input_ids = [tokenizer.cls_token_id] + \
                        [tokenizer.pad_token_id] * (seq_length - 2) + \
                        [tokenizer.sep_token_id]
        
        return torch.tensor([ref_input_ids])

    def lig_forward(self, input_ids):
        logits = self(input_ids=input_ids)[0]
        pred_prob = logits.unsqueeze(-1)
        return pred_prob

    def get_input_embeddings(self, input_ids):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return self.encoder.roberta.embeddings.word_embeddings(input_ids), attention_mask

    def get_vuln_prediction(self, inputs_ids=None, inputs_embeds=None, attention_mask=None):
        if inputs_embeds is not None:
            logits = self(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            logits = self(input_ids=inputs_ids)
        vuln_logits = logits[0]
        return int((vuln_logits > self.args.vuln_threshold).item()), vuln_logits