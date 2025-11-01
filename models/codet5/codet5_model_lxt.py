# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch

from models.base_explainable_model import ExplainableVulnerabilityModel

class DefectModelLXT(nn.Module, ExplainableVulnerabilityModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModelLXT, self).__init__()
        self.encoder = encoder 
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.args = args

    def get_t5_vec(self, source_ids, output_attentions=False, inputs_embeds=None):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)

        # This encoder requires both inputs_embeds and source_ids
        if inputs_embeds is not None and source_ids is not None:
            outputs = self.encoder(inputs_embeds=inputs_embeds,
                                   attention_mask=attention_mask,
                                   labels=source_ids,
                                   decoder_attention_mask=attention_mask,
                                   output_hidden_states=True,
                                   output_attentions=output_attentions)
        elif source_ids is not None:
            outputs = self.encoder(input_ids=source_ids,
                                   attention_mask=attention_mask,
                                   labels=source_ids, 
                                   decoder_attention_mask=attention_mask, 
                                   output_hidden_states=True,
                                   output_attentions=output_attentions)
        else:
            raise ValueError("Either source_ids or inputs_embeds must be provided.")
        
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

    def forward(self, input_ids=None, labels=None, weight=None, output_attentions=False, inputs_embeds=None):
        # input_ids = input_ids.view(-1, self.args.max_source_length)

        if self.args.model_type == 'codet5_patched' or self.args.model_type == 't5':
            if output_attentions:
                vec, attentions = self.get_t5_vec(input_ids, 
                                                  output_attentions=output_attentions,
                                                  inputs_embeds=inputs_embeds)
            else:
                vec = self.get_t5_vec(input_ids, 
                                      output_attentions=output_attentions,
                                      inputs_embeds=inputs_embeds)
        
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
        pass

    def get_reference_input_ids(self, input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
        pass

    def lig_forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor = None):
        pass

    def get_input_embeddings(self, input_ids):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return self.encoder.shared(input_ids), attention_mask

    def get_vuln_prediction(self, inputs_ids=None, inputs_embeds=None, attention_mask=None):
        logits = self(input_ids=inputs_ids, inputs_embeds=inputs_embeds)
        vuln_logits = logits.squeeze()[1]
        return int((vuln_logits > self.args.vuln_threshold).item()), vuln_logits