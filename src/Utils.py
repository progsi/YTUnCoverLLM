import os
import pandas as pd
import torch

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