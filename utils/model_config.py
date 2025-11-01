from models.linevul.linevul_model import LinevulModel
from models.codexglue.codexglue_model import CodexglueModel
from models.codet5.codet5_model import DefectModel
from transformers import (RobertaConfig, RobertaForSequenceClassification, 
                          RobertaTokenizer, T5Config, T5ForConditionalGeneration)

from models.base_models_patched.modeling_roberta_lxt import RobertaForSequenceClassification_LXT
from models.linevul.linevul_model_lxt import LinevulLXTModel
from models.base_models_patched.modeling_t5_lxt import T5ForConditionalGenerationLXT
from models.codet5.codet5_model_lxt import DefectModelLXT
from data_loaders import convert_examples_to_features_roberta, convert_examples_to_features_codet5
import logging

logger = logging.getLogger(__name__)

XAI_METHODS = ['attention', 'lig', 'attn_lrp']

MODEL_CLASSES = {
    'linevul':   ('microsoft/codebert-base', RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, convert_examples_to_features_roberta, LinevulModel),
    'codexglue': ('microsoft/codebert-base', RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, convert_examples_to_features_roberta, CodexglueModel),
    'codet5':    ('Salesforce/codet5-base', T5Config, T5ForConditionalGeneration, RobertaTokenizer, convert_examples_to_features_codet5, DefectModel),
    'linevul_patched': ('microsoft/codebert-base', RobertaConfig, RobertaForSequenceClassification_LXT, RobertaTokenizer, convert_examples_to_features_roberta, LinevulLXTModel),
    'codexglue_patched': ('microsoft/codebert-base', RobertaConfig, RobertaForSequenceClassification_LXT, RobertaTokenizer, convert_examples_to_features_roberta, CodexglueModel), # for this model, only the encoder changes
    'codet5_patched': ('Salesforce/codet5-base', T5Config, T5ForConditionalGenerationLXT, RobertaTokenizer, convert_examples_to_features_codet5, DefectModelLXT),
}

def load_model(args):
    if not args.model_type in MODEL_CLASSES:
        msg = f'ERROR --model_type wrong. Possibles model_type = {MODEL_CLASSES.keys()}'
        raise ValueError(msg)
    
    # Retrieve all the classes/functions needed
    model_path, config_class, model_class, tokenizer_class, convert_examples_to_features, ModelConstructor = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(model_path)
    config = config_class.from_pretrained(model_path)
    
    if args.model_type in ['linevul', 'linevul_patched', 'codexglue', 'codexglue_patched']:
        config.num_labels = 1

    encoder = model_class.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
    model = ModelConstructor(encoder, config, tokenizer, args)

    return model, tokenizer, convert_examples_to_features
