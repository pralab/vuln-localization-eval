import pickle
from torch.utils.data import DataLoader, SequentialSampler
import os
from tqdm import tqdm
import pandas as pd
import logging
from detection_alignment import DetectionAlignment
from .attribution_methods import create_explainer, BaseExplainer

logger = logging.getLogger(__name__)

def line_level_localization(explainer:BaseExplainer, tokenizer, model, mini_batch, args):
    (input_ids, _) = mini_batch
    ids = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(ids)
    all_tokens = [token.replace("Ġ", "") for token in all_tokens]
    all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]

    input_ids = input_ids.to(args.device)

    word_att_scores, all_lines_score_norm = explainer.get_scores(model=model,
                                                          input_ids=input_ids,
                                                          all_tokens=all_tokens,
                                                          tokenizer=tokenizer)

    return word_att_scores, all_lines_score_norm

def detection_alignment_fuzzy_iou(model, 
                                  tokenizer, 
                                  test_dataset,
                                  args):
    results = [] # Stores predictions for each sample
    test_sampler = SequentialSampler(test_dataset)

    xai_method = args.xai_method
    logger.info(f"**** xAI Method = {xai_method} ****")
    
    # Choose the proper explainer based on args
    try:
        explainer = create_explainer(args, model)
    except (ValueError, AttributeError) as e:
        logger.error(f"Error for xai method {xai_method}: {e}")
        return

    # Init DA metric
    det_alignment = DetectionAlignment()
    
    dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, num_workers=0)
    progress_bar = tqdm(dataloader, total=len(dataloader))

    model.eval()
    for index, mini_batch in enumerate(progress_bar):
        ground_truth = test_dataset.flaw_lines[index]

        if len(ground_truth) == 0:
            continue

        word_att_scores, all_lines_scores = line_level_localization(explainer=explainer,
                                                                    tokenizer=tokenizer,
                                                                    model=model,
                                                                    mini_batch=mini_batch,
                                                                    args=args)

        y_pred, _ = model.get_vuln_prediction(inputs_ids=mini_batch[0].to(args.device)) 

        det_alignment.update(lines=all_lines_scores["lines"],
                             scores=all_lines_scores["scores"],
                             y_pred=y_pred,
                             ground_truth=ground_truth)

        results.append({"all_lines_scores": all_lines_scores,
                        "word_attr_scores": word_att_scores,
                        "y_pred": y_pred,
                        "y_true": test_dataset.df.iloc[index].target,
                        "gt_sample": ground_truth})

    logger.info(f"Detection Alignment (DA) : {det_alignment.get_da():.8f}")
    
    # Save the results
    folder = "eval_results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(f"{folder}/{args.model_path.split('/')[1]}_{xai_method}_rel_scores.pkl", "wb") as f:
        pickle.dump(results, f)