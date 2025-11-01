import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification

from models.base_explainable_model import ExplainableVulnerabilityModel

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class LinevulModel(RobertaForSequenceClassification, ExplainableVulnerabilityModel):   
    def __init__(self, encoder, config, tokenizer, args):
        super(LinevulModel, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
       
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None, attention_mask=None):
        if attention_mask is None and input_ids is not None:
            attention_mask = input_ids.ne(1)

        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob
            
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
        y_pred = 1 # for positive attribution, y_pred = 0 for negative attribution
        pred_prob = logits[y_pred].unsqueeze(-1)
        return pred_prob

    def get_input_embeddings(self, input_ids: torch.Tensor):
        # implemented just for LXT models
        pass

    def get_vuln_prediction(self, inputs_ids):
        logits = self(input_ids=inputs_ids)
        vuln_logits = logits.squeeze()[1]
        return int((vuln_logits > self.args.vuln_threshold).item()), vuln_logits