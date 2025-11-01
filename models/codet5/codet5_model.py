# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch

from models.base_explainable_model import ExplainableVulnerabilityModel
     
class DefectModel(nn.Module, ExplainableVulnerabilityModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.args = args

    def get_t5_vec(self, source_ids, output_attentions=False):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True,
                               output_attentions=output_attentions)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            print(eos_mask.sum(1))
            print(torch.unique(eos_mask.sum(1)))
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        if output_attentions:
            return vec, outputs.encoder_attentions
        else:
            return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def forward(self, input_ids=None, labels=None, weight=None, output_attentions=False):
        # input_ids = input_ids.view(-1, self.args.max_source_length)

        if self.args.model_type == 'codet5' or self.args.model_type == 't5':
            if output_attentions:
                vec, attentions = self.get_t5_vec(input_ids, output_attentions=output_attentions)
            else:
                vec = self.get_t5_vec(input_ids, output_attentions=output_attentions)
        elif self.args.model_type == 'bart':
            vec = self.get_bart_vec(input_ids)
        elif self.args.model_type == 'roberta':
            vec = self.get_roberta_vec(input_ids)
        
        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            if output_attentions:
                return prob, attentions
            else:
                return prob
            
    def get_embedding_layer(self) -> nn.Module:
        return self.encoder.shared

    def get_reference_input_ids(self, input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
        seq_length = input_ids.size(1)
        ref_input_ids = [tokenizer.cls_token_id] + \
                        [tokenizer.pad_token_id] * (seq_length - 2) + \
                        [tokenizer.eos_token_id]
        
        return torch.tensor([ref_input_ids])

    def lig_forward(self, input_ids):
        logits = self(input_ids=input_ids)[0]
        y_pred = 1
        pred_prob = logits[y_pred].unsqueeze(-1)
        return pred_prob

    def get_input_embeddings(self, input_ids: torch.Tensor):
        # implemented just for LXT models
        pass
    
    def get_vuln_prediction(self, inputs_ids):
        with torch.no_grad():
            logits = self(input_ids=inputs_ids)
        vuln_logits = logits.squeeze()[1]
        return int((vuln_logits > self.args.vuln_threshold).item()), vuln_logits