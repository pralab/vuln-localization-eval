import torch
from abc import ABC, abstractmethod
from captum.attr import LayerIntegratedGradients

def clean_special_token_values(all_values, padding=False):
    # special token in the beginning of the seq 
    all_values[0] = 0
    if padding:
        # get the last non-zero value which represents the att score for </s> token
        idx = [index for index, item in enumerate(all_values) if item != 0][-1]
        all_values[idx] = 0
    else:
        # special token in the end of the seq 
        all_values[-1] = 0
    return all_values

def get_all_lines_score(tokenizer,
                        token_scores: list,
                        use_normalize: bool = False,
                        use_abs: bool = False) -> dict:
    all_lines_score = {"lines":[],"scores":[]}
    current_line_tokens = []
    line_score = 0

    for token, score in token_scores:
        token_reconstructed = tokenizer.convert_tokens_to_string(token)
        line_score += abs(score) if use_abs else score

        # Summerize if meet new line \n
        if token_reconstructed.strip("\n") == '':
            line_code = "".join(current_line_tokens)
            if line_code != "":
                all_lines_score["lines"].append(line_code)
                all_lines_score["scores"].append(line_score)

            current_line_tokens = []
            line_score = 0
        else:
            current_line_tokens.append(token_reconstructed)

    line_code = "".join(current_line_tokens)
    if line_code:
        all_lines_score["lines"].append(line_code)
        all_lines_score["scores"].append(line_score)

    assert len(all_lines_score["lines"]) == len(all_lines_score["scores"])

    # Normalize the scores between 0-1
    if use_normalize:
        scores_tensor = torch.tensor(all_lines_score["scores"], dtype=torch.float32)
        normalized_scores = (scores_tensor - scores_tensor.min()) / (scores_tensor.max() - scores_tensor.min() + 1e-8)
        all_lines_score["scores"] = normalized_scores.tolist()

    return all_lines_score

def get_word_att_scores(all_tokens: list, att_scores: list, tokenizer) -> list:
    special_tokens = set(tokenizer.all_special_tokens)
    return [
            (token, score) for token, score in zip(all_tokens, att_scores)
            if token not in special_tokens
            ]

def clean_word_attr_scores(word_attr_scores: list) -> list:
    to_be_cleaned = {'<s>', '</s>', '<unk>', '<pad>'}
    cleaned = []
    for word_attr_score in word_attr_scores:
        if word_attr_score[0] not in to_be_cleaned:
            cleaned.append(word_attr_score)
    return cleaned

class BaseExplainer(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def get_scores(self, model, input_ids, all_tokens, tokenizer):
        pass

class LIGExplainer(BaseExplainer):
    def __init__(self, args, lig_forward_func):
        super().__init__(args)
        self.lig_forward = lig_forward_func
    
    def summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        return attributions

    def get_scores(self, model, input_ids, all_tokens, tokenizer):
        ref_input_ids = model.get_reference_input_ids(input_ids, tokenizer)
        embeddings = model.get_embedding_layer()

        input_ids = input_ids.to(self.args.device)
        ref_input_ids = ref_input_ids.to(self.args.device)

        lig = LayerIntegratedGradients(self.lig_forward, embeddings)
        attributions, _ = lig.attribute(inputs=input_ids, 
                                        baselines=ref_input_ids,
                                        internal_batch_size=16,
                                        return_convergence_delta=True)
        
        attr_scores = self.summarize_attributions(attributions).tolist()
        assert len(all_tokens) == len(attr_scores)

        word_attr_scores = get_word_att_scores(all_tokens=all_tokens,
                                               att_scores=attr_scores,
                                               tokenizer=tokenizer)
        word_attr_scores = clean_word_attr_scores(word_attr_scores)

        return word_attr_scores, get_all_lines_score(tokenizer=tokenizer,
                                                    token_scores=word_attr_scores, 
                                                    use_normalize=self.args.use_normalize_attention_scores, 
                                                    use_abs=self.args.use_abs_attention_scores)

class AttentionExplainer(BaseExplainer):
    def __init__(self, args, block_index=-1):
        super().__init__(args)
        self.block_index = block_index
    
    def get_scores(self, model, input_ids, all_tokens, tokenizer):
        model.eval()
        model.to(self.args.device)
        
        with torch.no_grad():
            _, attentions = model(input_ids=input_ids, output_attentions=True)

        attentions = attentions[self.block_index][0]
        attention = None

        for i in range(len(attentions)):
            layer_attention = attentions[i].sum(dim=0) 
            if attention is None:
                attention = layer_attention
            else:
                attention += layer_attention

        attention = clean_special_token_values(attention, padding=True)
        word_att_scores = get_word_att_scores(all_tokens=all_tokens, 
                                            att_scores=attention,
                                            tokenizer=tokenizer)
        
        return word_att_scores, get_all_lines_score(tokenizer=tokenizer,
                                                    token_scores=word_att_scores,
                                                    use_normalize=self.args.use_normalize_attention_scores,
                                                    use_abs=self.args.use_abs_attention_scores)

class AttnLRPExplainer(BaseExplainer):
    def __init__(self, args):
        super().__init__(args)

    def get_scores(self, model, input_ids, all_tokens, tokenizer):
        input_embeds, att_mask = model.get_input_embeddings(input_ids)

        # Activate gradient tracking
        input_embeds.requires_grad_()
        input_embeds.retain_grad()

        _, output_prob = model.get_vuln_prediction(inputs_ids=input_ids,
                                                inputs_embeds=input_embeds,
                                                attention_mask=att_mask)

        output_prob.backward()

        # Get the relevance
        relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]
        relevance = relevance / relevance.abs().max()

        word_attr_scores = get_word_att_scores(all_tokens=all_tokens, 
                                            att_scores=relevance,
                                            tokenizer=tokenizer)
        word_attr_scores = clean_word_attr_scores(word_attr_scores)

        x = get_all_lines_score(tokenizer=tokenizer,
                                token_scores=word_attr_scores,
                                use_normalize=self.args.use_normalize_attention_scores,
                                use_abs=self.args.use_abs_attention_scores)
        return word_attr_scores, x

def create_explainer(args, model):
    method = args.xai_method

    if method == "attention":
        block_index = getattr(args, 'block_index', -1) 
        return AttentionExplainer(args, block_index=block_index)
        
    elif method == "lig":
        if not hasattr(model, 'lig_forward'):
            raise AttributeError(f"The model {model.__class__.__name__} doesn't have the 'lig_forward' method required for LIG explainer.")
        
        return LIGExplainer(args, lig_forward_func=model.lig_forward)
        
    elif method == "attn_lrp":
        return AttnLRPExplainer(args)
        
    else:
        raise ValueError(f"Unknown method: {method}")