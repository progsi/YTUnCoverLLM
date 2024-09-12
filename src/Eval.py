
import pandas as pd
import os
import io
from typing import Tuple, List, Dict
import numpy as np
from src.Utils import read_IOB_file, parse_preds, read_jsonlines, clean_dict
from preprocessing.Utils import make_taglist, retag_matches, SONG_ATTRS, simplify_string
import importlib
import sys
sys.path.append('../baseline/music-ner-eacl2023/music-ner/src')
eval_utils = importlib.import_module("eval_utils")
from eval_utils import compute_results
from tqdm import tqdm
from contextlib import redirect_stdout


def __load_IOB_labels(filepath: str) -> Tuple[List[np.array], List[np.array]]:
    texts, IOBs = read_IOB_file(filepath)
    yt_processed = [' '.join(s) for s in texts]
    return yt_processed, IOBs

def __match_data(data: pd.DataFrame) -> pd.DataFrame:
    mask = data.IOB_pred.apply(len) != data.IOB.apply(len)
    print(f"INFO: {len(data[mask])} non matching IOB taglists.")
    matching_data = data[~mask]
    return matching_data

def jsonl_to_dataframe(jsonl_filepath: str, iob_filepath: str) -> pd.DataFrame:
    """
    Args:
        jsonl_filepath (str): 
        iob_filepath (str): 
    Returns:
        pd.DataFrame: 
    """
    data = pd.DataFrame(parse_preds(jsonl_filepath), columns=["performer", "title"])

    data.performer_processed = data.performer.str.replace(";", " feat. ")

    yt_processed, IOBs = __load_IOB_labels(iob_filepath)
    data["IOB_true"] = IOBs
    data["yt_processed"] = yt_processed
    return data

def compute_results_jsonl(iob_filepath: str, jsonl_filepath: str, min_r: int = 100) -> pd.DataFrame:
    """Compute the results from predictions in a jsonlines file.
    Args:
        iob_filepath (str): filepath to dataset with IOB true labels
        jsonl_filepath (str): filepath to predictions of extracted attributes
        min_r (int): minimum ratio for matching of LLM-extracted attributes to input text
    Returns:
        pd.DataFrame: with the matching IOB taglists
    """
    # load jsonl predictions
    data = pd.DataFrame(parse_preds(jsonl_filepath), columns=["performer_processed", "title_processed"])
    data.performer_processed = data.performer_processed.apply(lambda x: x.split(";"))

    # load true labels
    yt_processed, IOBs = __load_IOB_labels(iob_filepath)
    data["yt_processed"] = yt_processed
    data["IOB"] = IOBs
    
    tqdm.pandas()
    data["TEXT"] = data.yt_processed.progress_apply(lambda x: simplify_string(x).split())

    # annotate based on min_r
    ent_names = [ent + '_processed' for ent in SONG_ATTRS if ent + '_processed' in data.columns]
    series = data.progress_apply(make_taglist, args=(ent_names, True, True, min_r), axis=1)
    data["IOB_pred"] = series.apply(lambda x: x[0])
    data["IOB_Indel_llm"] = series.apply(lambda x: x[1])

    print("Retag IOBs...")
    data["IOB_pred"] = data.progress_apply(lambda x: retag_matches(x.TEXT, x.IOB_pred), axis=1)


    # filter non-matching rows
    matching_data = __match_data(data)
    compute_results(matching_data.IOB.to_list(), matching_data.IOB_pred.to_list())
    return matching_data

def compute_results_txt(iob_filepath: str, pred_filepath: str) -> pd.DataFrame:
    """Compute results based on two bio files
    Args:
        iob_filepath (str): true labels in .bio file
        pred_filepath (str): prediction textfile
    Returns:
        pd.DataFrame: with the matching IOB taglists
    """
    # load true labels
    yt_processed, IOBs_true = __load_IOB_labels(iob_filepath)
    IOBs_pred = parse_preds(pred_filepath)
        
    data = pd.DataFrame(zip(IOBs_true, IOBs_pred), columns=["IOB", "IOB_pred"])
    data["yt_processed"] = yt_processed

    matching_data = __match_data(data)
    compute_results(matching_data.IOB.to_list(), matching_data.IOB_pred.to_list())
    return matching_data

def eval_llm(jsonl_path: str) -> None:
    """Eval the LLM.
    Args:
        jsonl_path (str): path to output jsonl file
    """
    data = read_jsonlines(jsonl_path)
    print(f"Input path: {jsonl_path}")
    true_iobs, pred_iobs = get_iobs_from_data(data)
    return compute_results(true_iobs, pred_iobs)

def get_iobs_from_data(data: List[dict]) -> Tuple[List[str], List[str]]:
    def get_taglist_true(item: Dict) -> List[str]:
        """Get true taglist based on pred item.
        Args:
            item (Dict): 
        Returns:
            List[str]: tag list IOB
        """
        item["title"] = item["titles"]
        item["performer"] = item["performers"]
        iobs, _ = make_taglist(item, ent_names=["title", "performer"], baseline_name=True, all=True, min_r=100, text_col="text")
        return iobs

    def get_taglist_pred(item: Dict) -> List[str]:
        """Get prediction taglist based on pred item.
        Args:
            item (Dict): 
        Returns:
            List[str]: tag list IOB
        """
        ent_list = item["extracted"]
        if isinstance(ent_list, list) and len(ent_list) > 0 and ent_list[0].get("content"):
            ent_list = ent_list[0].get("content")
        elif isinstance(ent_list, dict):
            ent_list = [v for (k,v) in ent_list.items()]
        for label in ["title", "performer"]:
            item[label] = [e["utterance"] for e in ent_list if isinstance(e, dict) and e.get("label") and e.get("label").lower() == label]
        iobs, _ = make_taglist(item, ent_names=["title", "performer"], baseline_name=True, all=True, min_r=100, text_col="text")
        return iobs
    
    true_iobs = []
    pred_iobs = []

    for item in data:
        true_iobs.append(get_taglist_true(item))
        pred_iobs.append(get_taglist_pred(item))
    return true_iobs, pred_iobs


def parse_filename(filename):
    parts = filename.replace(".jsonl", "").split("_")
    dataset = parts[0]
    k_shot = parts[1].split(".")[0]
    k = int(k_shot.replace("shot", ""))
    if k and k != 0 and len(parts) == 3:
        sampling_method = parts[2]
    else:
        sampling_method = "rand"
    return dataset, k, sampling_method

def silent_eval_llm(fpath):
    with io.StringIO() as buf, redirect_stdout(buf):
        # Call the actual eval_llm function
        return eval_llm(fpath)
    
def aggregated_results_llm(
    results_dir,
    ent_types=["overall", "Artist", "WoA"],
    eval_schemas=["strict", "exact", "ent_type"],
    metrics=["f1", "f1_macro", "f1_micro", "precision", "precision_macro", "recall", "recall_macro", "missed", "spurious", "incorrect"],
    datasets=["dataset1", "dataset2", "dataset3", "dataset4", "dataset5"],
    sampling_methods=["rand", "tfidf", ""],
    ks=[0,5,15,25,35,45]
):
    results = {}
    for schema in eval_schemas:
        results[schema] = {}
        for ent_type in ent_types:
            results[schema][ent_type] = {}
            for metric in metrics:
                results[schema][ent_type][metric] = {}
                for model in [m for m in os.listdir(results_dir) if m != "archive"]:
                    results[schema][ent_type][metric][model] = {}
                    for sampling_method in sampling_methods:
                        results[schema][ent_type][metric][model][sampling_method] = {}
                        for k in ks:
                            results[schema][ent_type][metric][model][sampling_method][k] = []

    model_dirs = [
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d != "archive"
    ]

    for model_dir in model_dirs:
        model = model_dir.split(os.sep)[-1]
        files = [os.path.join(dirpath, filename) 
             for dirpath, _, filenames in os.walk(model_dir) 
             for filename in filenames]

        for fpath in files:
            print(fpath)
            filename = os.path.basename(fpath)
            dataset, k, sampling_method = parse_filename(filename)

            if dataset in datasets and k in ks:
                predictions = silent_eval_llm(fpath)

                for ent_type in ent_types:
                    for schema in eval_schemas:
                        for metric in metrics:
                            key = f"{ent_type}_{schema}_{metric}"
                            if key in predictions.keys():
                                results[schema][ent_type][metric][model][sampling_method][k].append(predictions[key])
                                if k == 0:
                                    for other_sampling_method in [s for s in sampling_methods if s != sampling_method]:
                                        results[schema][ent_type][metric][model][other_sampling_method][k].append(predictions[key])
    return clean_dict(results)

def results_to_dataframe(aggregated_results, agg_func='mean'):
    data = []

    for schema, schema_dict in aggregated_results.items():
        for ent_type, ent_type_dict in schema_dict.items():
            for metric, metric_dict in ent_type_dict.items():
                for model, model_dict in metric_dict.items():
                    for sampling, sampling_dict in model_dict.items():
                        for k_shot, values in sampling_dict.items():
                            if values:
                                if agg_func == 'mean':
                                    agg_value = np.mean(values)
                                elif agg_func == 'sum':
                                    agg_value = np.sum(values)
                                else:
                                    raise ValueError("Invalid aggregation function. Use 'mean' or 'sum'.")
                                
                                data.append([schema, ent_type, metric, model, sampling, k_shot, agg_value])

    df = pd.DataFrame(data, columns=['Schema', 'Entity Type', 'Metric', 'Model', 'Sampling', 'k', 'Value'])
    df.Model = df.Model.str.replace("mistral", "Mistral-7B").str.replace("mixtral", "Mixtral-8x22B").str.replace("gpt-3.5-turbo-0125", "GPT-3.5-Turbo")
    return df.set_index(['Schema', 'Entity Type', 'Metric', 'Model', 'Sampling', 'k']).unstack(['Schema', 'Entity Type', 'Metric'])

def results_for_metric(aggregated_results, metric):
    data = results_to_dataframe(aggregated_results, agg_func='mean')
    return data[[   ('Value',   'strict', 'overall', f'{metric}_macro'), 
                    ('Value',   'strict',     'WoA', metric),
                    ('Value',   'strict',  'Artist', metric)]]