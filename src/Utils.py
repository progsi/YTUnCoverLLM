import os
import json
import pandas as pd
import numpy as np
import torch
from typing import Tuple, List, Dict
from preprocessing.Utils import overlap
from baseline.parse_output import parse_pred_file


def read_textfile(path: str):

    with open(path, "r") as f:
        content = f.read()
    return content

def read_jsonfile(path: str):

    with open(path, "r") as f:
        content = json.load(f)
    return content

def read_jsonlines(file_path: str) -> List[str]:
    """
    Reads a JSON Lines (jsonl) file and returns a list of JSON objects.
    Parameters:
        file_path (str): The path to the JSON Lines file.
    Returns:
         List[str]: A list of JSON objects.
    """
    json_objects = []
    with open(file_path, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            json_objects.append(json_object)
    return json_objects

def get_key(service: str):

    return read_textfile(os.path.join("keys", f"{service}.txt"))

def get_target_matrix(data: pd.DataFrame):
    """Generates the binary square matrix of cover song relationships between all
    elements in the dataset.
    Returns:
        np.array: binary square matrix
    """
    set_ids = data["set_id"].values
    target = (set_ids[:, None] == set_ids)
    return torch.from_numpy(target).to(dtype=torch.int)

def get_concat(df: pd.DataFrame, attrs: List[str]) -> List[str]:
    """Get concated list of strings from dataframe.
    Args:
        df (pd.DataFrame): input dataframe
        attrs (List[str]): which attributes to consider
    Returns:
        List[str]: concated strings in list
    """
    return df[attrs].apply(lambda row: ' '.join(map(str, row)), axis=1).to_list()

def get_left_right_concat(df: pd.DataFrame, left_attrs: List[str], right_attrs: List[str]) -> Tuple[List[str],List[str]]:
    """Helper to get left and right concatenated as list.
    Args:
        df (pd.DataFrame): DataFrame with metadata
        left_attrs (List[str]): attributes of left side
        right_attrs (List[str]): attributes of right side.
    Returns:
        List[str], List[str]: left and right data strings.
    """
    left_data = get_concat(df, left_attrs)
    right_data =  get_concat(df, right_attrs)
    
    return left_data, right_data

def get_concat_col_name(left_attrs: List[str], right_attrs: List[str]) -> Tuple[List[str], List[str]]:
    """Generate column name for the col with the pairwise similarity of concatenated columns 
    (eg. entity-level).
    Args:
        left_attrs (List[str]): attributes on the left side
        right_attrs (List[str]): attributes on the right side
    """
    return ('+'.join([attr for attr in left_attrs]), '+'.join([attr for attr in right_attrs]))  


def get_ents_spans(IOB: List[str]) -> Dict[Tuple[int, int], str]:
    """Get ent spans from a list of IOB tags.
    Args:
        IOB (List[str]): IOB tags
    Returns:
        Dict[Tuple[int, int], str]: mapping of ent spans
    """
    ents = {}
    cur_type = None
    start = None

    for i, token in enumerate(IOB):
        if token.startswith('B-'):
            if cur_type:
                ents.setdefault(cur_type, []).append((start, i - 1))
            cur_type = token[2:]
            start = i
        elif token.startswith('I-') and cur_type == token[2:]:
            continue
        else:
            if cur_type:
                ents.setdefault(cur_type, []).append((start, i - 1))
                cur_type = None

    if cur_type:
        ents.setdefault(cur_type, []).append((start, len(IOB) - 1))

    return ents

def get_missing_ents(IOB_true: List[str], IOB_pred: List[str]) -> Dict[Tuple[int, int], str]:
    """Get missing entities based on true and pred IOBs.
    Args:
        IOB_true (List[str]): 
        IOB_pred (List[str]): 
    Returns:
        Dict[Tuple[int, int], str]: mapping of missing entities by spans
    """
    ents_true, ents_pred = get_ents_spans(IOB_true), get_ents_spans(IOB_pred)

    missing_ents = {}
    for ent, spans_true in ents_true.items():
        for span_true in spans_true:
            spans_pred = ents_pred.get(ent)
            if not spans_pred or not any([overlap(span_true, span_pred) for span_pred in spans_pred]):
                missing_ents[span_true] = ent
    return missing_ents

def get_spurious_ents(IOB_true: List[str], IOB_pred: List[str]) -> Dict[Tuple[int, int], str]:
    """Get spurious entities based on true and pred IOBs.
    Args:
        IOB_true (List[str]): 
        IOB_pred (List[str]): 
    Returns:
        Dict[Tuple[int, int], str]: mapping of missing entities by spans
    """
    ents_true, ents_pred = get_ents_spans(IOB_true), get_ents_spans(IOB_pred)

    spurious_ents = {}
    for ent, spans_pred in ents_pred.items():
        for span_pred in spans_pred:
            spans_true = ents_true.get(ent)
            if not spans_true or not any([overlap(span_true, span_pred) for span_true in spans_true]):
                spurious_ents[span_pred] = ent
    return spurious_ents
         
def get_incorrect_ents(IOB_true: List[str], IOB_pred: List[str]) -> Dict[Tuple[int, int], str]:
    """Get incorrect entities based on true and pred IOBs.
    Args:
        IOB_true (List[str]): 
        IOB_pred (List[str]): 
    Returns:
        Dict[Tuple[int, int], str]: mapping of incorrect entities by spans
    """
    ents_true, ents_pred = get_ents_spans(IOB_true), get_ents_spans(IOB_pred)

    incorrect_ents = {}
    for ent, spans_true in ents_true.items():
        for span_true in spans_true:
            spans_pred = ents_pred.get(ent)
            if spans_pred:
                for span_pred in ents_pred[ent]:
                    if span_pred != span_true and overlap(span_true, span_pred): 
                        incorrect_ents[span_pred] = ent
    return incorrect_ents

def get_error_analysis(data: pd.DataFrame, pred_col: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    series_missing = data.apply(lambda row: get_missing_ents(row["IOB"], row[pred_col]), axis=1)
    series_spurious = data.apply(lambda row: get_spurious_ents(row["IOB"], row[pred_col]), axis=1)
    series_incorrect = data.apply(lambda row: get_incorrect_ents(row["IOB"], row[pred_col]), axis=1)
    return series_missing, series_spurious, series_incorrect

def read_IOB_file(path: str) -> Tuple[List[np.array], List[np.array]]:
    """Read in an IOB textfile
    Args:
        path (str): path to IOB file
    Returns:
        Tuple[List[np.array], List[np.array]]: list of texts and list of IOBs
    """
    with open(path, "r") as f:
        content = f.readlines()
    
    cur_words = []
    cur_tags = []
    words = []
    tags = []
    for row in content:
        if row == "\n":
            words.append(np.array(cur_words, dtype="object"))
            tags.append(np.array(cur_tags, dtype="object"))
            cur_words = []
            cur_tags = []
        else:
            row = row.replace("\n", "").split("\t")
            cur_words.append(row[0])
            cur_tags.append(row[1])
    return words, tags

def parse_preds(file_path: str) -> List[np.array]:
    """Parse predictions flexibally. File can be a jsonl or a textfile.
    Args:
        file_path (str): 
    Returns:
        List[np.array]: 
    """
    if file_path.endswith(".jsonl"):
        content = read_jsonlines(file_path)
        return [np.array(list(d.values()), dtype="object") for d in content]
    else:
        return parse_pred_file(file_path)


