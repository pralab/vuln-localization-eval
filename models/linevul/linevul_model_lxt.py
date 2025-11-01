import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

#from ..base_models_patched import RobertaForSequenceClassification_LXT, LXTDropout
from ..base_models_patched import RobertaForSequenceClassification_LXT
from ..base_models_patched import LXTDropout
from lxt.efficient.rules import identity_rule_implicit

from models.base_explainable_model import ExplainableVulnerabilityModel

class RobertaClassificationHead(nn.Module, ExplainableVulnerabilityModel):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = LXTDropout(classifier_dropout)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = identity_rule_implicit(self.activation, x) ### <------------------------------------------- LXT
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class LinevulLXTModel(RobertaForSequenceClassification_LXT):   
    def __init__(self, encoder, config, tokenizer, args):
        super(LinevulLXTModel, self).__init__(config=config)
        self.tokenizer = tokenizer
        config.output_attentions = True

        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.encoder = encoder
       
    def forward(self, inputs_embeds=None, labels=None, output_attentions=False, input_ids=None, attention_mask=None):
        if attention_mask is None and (input_ids is not None or inputs_embeds is not None):
            attention_mask = input_ids.ne(1)

        if output_attentions:

            outputs = self.encoder.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                inputs_embeds=inputs_embeds,
                return_dict=True 
            )
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            outputs = self.encoder.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                inputs_embeds=inputs_embeds,
            )[0]
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
                return loss, prob
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
        return self.encoder.roberta.embeddings.word_embeddings(input_ids), attention_mask
    
    def get_vuln_prediction(self, inputs_ids=None, inputs_embeds=None, attention_mask=None):
        if inputs_embeds is not None:
            logits = self(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            logits = self(input_ids=inputs_ids)
        vuln_logits = logits.squeeze()[1]
        return int((vuln_logits > self.args.vuln_threshold).item()), vuln_logits