import logging
import random
import numpy as np
import torch

from data_loaders import TextDataset
from eval_utils import detection_alignment_fuzzy_iou
from utils import args_parser, load_model

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def line_prediction_iou_eval():
    args = args_parser()
    set_seed(args)
    
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.info("Running experiments on device: %s", args.device)

    logger.info("Training/evaluation parameters %s", args)
    model, tokenizer, convert_examples_to_features = load_model(args)
    
    model.load_state_dict(torch.load(args.model_path, map_location=args.device), strict=False)
    model.to(args.device)
    logger.info("Loaded pretrained model")
        
    logger.info("***** Running DA Line Evaluation *****")
    test_dataset = TextDataset(tokenizer, 
                               args, 
                               file_type='test', 
                               only_vulnerable=True, 
                               convert_examples_to_features=convert_examples_to_features)
    
    detection_alignment_fuzzy_iou(args=args, 
                                  model=model, 
                                  tokenizer=tokenizer, 
                                  test_dataset=test_dataset)

if __name__ == "__main__":
    line_prediction_iou_eval()