
import pandas as pd
from src.Utils import read_IOB_file, parse_preds, read_jsonlines
from preprocessing.Utils import make_taglist, retag_matches, SONG_ATTRS, simplify_string
from typing import Tuple, List, Dict
import numpy as np
import importlib
import sys
sys.path.append('../baseline/music-ner-eacl2023/music-ner/src')
eval_utils = importlib.import_module("eval_utils")
from eval_utils import compute_results
from tqdm import tqdm


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
        if len(ent_list) > 0 and ent_list[0].get("content"):
            ent_list = ent_list[0].get("content")
        for label in ["title", "performer"]:
            item[label] = [e["utterance"] for e in ent_list if isinstance(e, dict) and e.get("label") and e.get("label").lower() == label]
        iobs, _ = make_taglist(item, ent_names=["title", "performer"], baseline_name=True, all=True, min_r=100, text_col="text")
        return iobs
    
    true_iobs = []
    pred_iobs = []

    for item in data:
        true_iobs.append(get_taglist_true(item))
        pred_iobs.append(get_taglist_pred(item))
    
    print(f"Input path: {jsonl_path}")
    return compute_results(true_iobs, pred_iobs)

    