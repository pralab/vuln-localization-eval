import argparse
from utils.model_config import MODEL_CLASSES, XAI_METHODS

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_file", default="/your/dataset/path", type=str, help="Dataset path.")
    parser.add_argument("--model_path", default="/your/pretrained/model/path", type=str, help="Model binary file path.")
    parser.add_argument("--model_type", default="", type=str, help="Models available: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--block_size", default=512, type=int, help="Input sequence length after tokenization.")
    parser.add_argument('--block_index', type=int, default=0, help="Transformer block number for attention eval")
    parser.add_argument("--device", default="cpu", help="Device (cuda:n or cpu)")
    parser.add_argument("--xai_method", default="", type=str, help="xAI methods available: " + ", ".join(XAI_METHODS))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--vuln_threshold", type=float, default=0.5, help='Vulnerability threshold')
    parser.add_argument("--use_normalize_attention_scores", action='store_true', default=True, help="Whether to normalize relevance scores.")
    parser.add_argument("--use_abs_attention_scores", action='store_true', default=False, help="Whether to apply abs to relevance scores.")

    return parser.parse_args()