from rapidfuzz import process, fuzz
import torch
import pandas as pd
from typing import List, Callable, Union, Tuple
import numpy as np
from tqdm import tqdm
from src.Utils import get_concat_col_name, get_left_right_concat


class Matcher(object):
    """Base class to structure the matching process with DataFrames as input.
    """
    def __init__(self, func: Union[Callable[[str, str], float], str], workers: int = 64) -> None:
        self.func = eval(func) if isinstance(func, str) else func
        self.workers = workers

    def match_square(self, df: pd.DataFrame, left_attrs: List[str], right_attrs: List[str]) -> pd.DataFrame:
        """Match square-wise on Entity-level (concatenated attributes)
        Args:
            df (pd.DataFrame): pandas Dataframe including the metadata
            left_attrs (List[str]): columns of the left side
            right_attrs (List[str]): columns of the right side
        Returns:
            pd.DataFrame: output dataframe
        """
        left_data, right_data = get_left_right_concat(df, left_attrs, right_attrs)
        y = self._match_square(left_data, right_data)
        return pd.DataFrame(y, index=df[left_attrs].index, columns=df[right_attrs].index)
    
    def match_pairwise(self, df: pd.DataFrame, left_attrs: List[str], right_attrs: List[str], 
                       level: str) -> pd.DataFrame:
        """Match all pairs as in df.
        Args:
            df (pd.DataFrame): pandas Dataframe including the metadata
            left_attrs (List[str]): columns of the left side
            right_attrs (List[str]): columns of the right side
            level (str): can be "entity" or "attribute". 
                "entity" --> match all attributes concatenated
                "attr" --> match attributes individually 
        Returns:
            pd.DataFrame: output dataframe
        """
        assert level in ["entity", "attr"], f"Invalid level {level}"

        if level == "entity":

            left_data, right_data = get_left_right_concat(df, left_attrs, right_attrs)
            col_name = get_concat_col_name(left_attrs, right_attrs)
            df[col_name] = self._match_pairwise(left_data, right_data)

        elif level == "attr":
            combis = pd.MultiIndex.from_product([left_attrs, right_attrs],  
                                                names=['left_attrs', 'right_attrs'])

            for combi in combis:
                
                left_attr, right_attrs = combi
                df[combi] = self._match_pairwise(df[left_attr], df[right_attrs])

        return df  
    
    def _match_pairwise(self, left_data: List[str], right_data: List[str]) -> np.array:
        return process.cpdist(left_data, right_data, scorer=self.func, workers=self.workers)
    
    def _match_square(self, left_data: List[str], right_data: List[str])-> np.array:
        return process.cdist(left_data, right_data, scorer=self.func, workers=self.workers)

