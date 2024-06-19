
import pandas as pd
from src.Utils import read_IOB_file, parse_preds
from preprocessing.Utils import make_taglist, SONG_ATTRS
from typing import Tuple, List
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

def compute_results_jsonl(iob_filepath: str, jsonl_filepath: str, min_r: int = 100):
    """Compute the results from predictions in a jsonlines file.
    Args:
        iob_filepath (str): filepath to dataset with IOB true labels
        jsonl_filepath (str): filepath to predictions of extracted attributes
        min_r (int): minimum ratio for matching of LLM-extracted attributes to input text
    """
    # load jsonl predictions
    data = pd.DataFrame(parse_preds(jsonl_filepath), columns=["performer_processed", "title_processed"])
    data.performer_processed = data.performer_processed.apply(lambda x: x.split(";"))

    # load true labels
    yt_processed, IOBs = __load_IOB_labels(iob_filepath)
    data["yt_processed"] = yt_processed
    data["IOB"] = IOBs

    # annotate based on min_r
    ent_names = [ent + '_processed' for ent in SONG_ATTRS if ent + '_processed' in data.columns]
    tqdm.pandas()
    series = data.progress_apply(make_taglist, args=(ent_names, True, True, min_r), axis=1)
    data["IOB_pred"] = series.apply(lambda x: x[0])
    data["IOB_Indel_llm"] = series.apply(lambda x: x[1])

    # filter non-matching rows
    matching_data = __match_data(data)
    compute_results(matching_data.IOB.to_list(), matching_data.IOB_pred.to_list())

def compute_results_txt(iob_filepath: str, pred_filepath: str):
    """Compute results based on two bio files
    Args:
        iob_filepath (str): true labels in .bio file
        pred_filepath (str): prediction textfile
    """
    # load true labels
    _, IOBs_true = __load_IOB_labels(iob_filepath)
    IOBs_pred = parse_preds(pred_filepath)

    data = pd.DataFrame(zip(IOBs_true, IOBs_pred), columns=["IOB", "IOB_pred"])
    matching_data = __match_data(data)
    compute_results(matching_data.IOB.to_list(), matching_data.IOB_pred.to_list())
