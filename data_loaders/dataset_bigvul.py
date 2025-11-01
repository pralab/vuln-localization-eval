"""
Bigvul Torch Dataset loader
"""
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
        
def convert_examples_to_features_roberta(func, label, tokenizer, args):
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label)

def convert_examples_to_features_codet5(func, label, tokenizer, args):
    if "</s>" in func:
        func = func.split("</s>")[0]

    source_ids = tokenizer.encode(func, max_length=args.block_size, padding='max_length', truncation=True)
    source_tokens = tokenizer.convert_ids_to_tokens(source_ids)
    return InputFeatures(source_tokens, source_ids, label)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 token_line_mapping=None):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.token_line_mapping = token_line_mapping

class TextDataset(Dataset):
    BIGVUL_FLAW_LINE_SEPARATOR = '/~/'

    def __init__(self, 
                 tokenizer, 
                 args, 
                 file_type="train", 
                 only_vulnerable=False, 
                 convert_examples_to_features=convert_examples_to_features_roberta
                 ):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        self.df = pd.read_csv(file_path)
        
        if only_vulnerable:
            self.df = self.df[self.df["target"]==1]
        
        funcs = self.df["processed_func"].tolist()
        labels = self.df["target"].tolist()
        self.flaw_lines = self._prepare_vulnerable_lines(self.df["flaw_line"].tolist())

        for i in tqdm(range(len(funcs))):
            self.examples.append(convert_examples_to_features(funcs[i], labels[i], tokenizer, args))
        
    def _prepare_vulnerable_lines(self, source_flaw_lines: list):
        '''
            Receives a list of flaw lines, in which the ith 
            element contains the set of vulnerable 
            strings for the ith sample
        '''
        res = []

        for lines in source_flaw_lines:
            set_flaw_lines = set()
            
            if isinstance(lines, str):
                lines_splitted = lines.split(self.BIGVUL_FLAW_LINE_SEPARATOR)
                #print(lines_splitted)
                for line in lines_splitted:
                    # Normalize the line (strip whitespaces and remove spaces)
                    set_flaw_lines.add(line.strip().replace(" ",""))

            res.append(set_flaw_lines) 
        return res

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)