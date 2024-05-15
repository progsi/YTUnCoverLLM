import os
import pandas as pd
import torch
from typing import Tuple, List


def read_textfile(path: str):

    with open(path, "r") as f:
        content = f.read()
    return content

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