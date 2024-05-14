from rapidfuzz import process, fuzz
import torch
import pandas as pd
from typing import List, Callable
from abc import abstractmethod, ABC
import numpy as np


class Matcher(ABC):
    """Base class to structure the matching process with DataFrames as input.
    """
    def match_square(self, df: pd.DataFrame, left_attrs: List[str], right_attrs: List[str]):
        """Match square-wise on Entity-level (concatenated attributes)
        Args:
            df (pd.DataFrame): pandas Dataframe including the metadata
            left_attrs (List[str]): columns of the left side
            right_attrs (List[str]): columns of the right side
        Returns:
            df: pd.DataFrame
        """
        
        left_data, right_data = self.__get_left_right_concat(df, left_attrs, right_attrs)
        y = self._match_square(left_data, right_data)
        return pd.DataFrame(y, index=df[left_attrs].index, columns=df[right_attrs].index)
    
    def match_pairwise(self, df: pd.DataFrame, left_attrs: List[str], right_attrs: List[str], 
                       level: str):
        """Match all pairs as in df.
        Args:
            df (pd.DataFrame): pandas Dataframe including the metadata
            left_attrs (List[str]): columns of the left side
            right_attrs (List[str]): columns of the right side
            level (str): can be "entity" or "attribute". 
                "entity" --> match all attributes concatenated
                "attr" --> match attributes individually 
        Returns:
            _type_: _description_
        """

        assert level in ["entity", "attr"], f"Invalid level {level}"

        if level == "entity":

            left_data, right_data = self.__get_left_right_concat(df, left_attrs, right_attrs)
            col_name = self.__get_concat_col_name(left_attrs, right_attrs)
            df[col_name] = self._match_pairwise(left_data, right_data)

        elif level == "attr":
            combis = pd.MultiIndex.from_product([left_attrs, right_attrs],  
                                                names=['left_attrs', 'right_attrs'])

            for combi in combis:
                
                left_attr, right_attrs = combi
                df[combi] = self._match_pairwise(df[left_attr], df[right_attrs])

        return df
    
    @staticmethod
    def __get_left_right_concat(df: pd.DataFrame, left_attrs: List[str], right_attrs: List[str]):
        """Helper to get left and right concatenated as list.
        Args:
            df (pd.DataFrame): DataFrame with metadata
            left_attrs (List[str]): attributes of left side
            right_attrs (List[str]): attributes of right side.
        Returns:
            List[str], List[str]: left and right data strings.
        """
        
        left_data = df[left_attrs].apply(lambda row: ' '.join(map(str, row)), axis=1)
        right_data = df[right_attrs].apply(lambda row: ' '.join(map(str, row)), axis=1)
        
        return left_data, right_data
    
    @staticmethod
    def __get_concat_col_name(left_attrs: List[str], right_attrs: List[str]):
        """Generate column name for the col with the pairwise similarity of concatenated columns 
        (eg. entity-level).
        Args:
            left_attrs (List[str]): attributes on the left side
            right_attrs (List[str]): attributes on the right side
        """
        return ('+'.join([attr for attr in left_attrs]), '+'.join([attr for attr in right_attrs]))    
    
    @abstractmethod
    def _match_square(left_data: List[str], right_data: List[str]):
        """Match all N^2 pairs of items in left and right data.
        Args:
            left_data (List[str]): left strings 
            right_data (List[str]): right strings
        """
        pass

    @abstractmethod
    def _match_pairwise(left_data: List[str], right_data: List[str]):
        """Match all pairs of strings in left and right data. Only strings at the 
        matching indices are matched.
        Args:
            left_data (List[str]): left strings 
            right_data (List[str]): right strings
        """
        pass

class FuzzyMatcher(Matcher):
    def __init__(self, func_str: str, workers: int = 64) -> None:
        self.func = self._func_from_str(func_str)
        self.workers = workers

    def _match_pairwise(self, left_data: List[str], right_data: List[str]):
        return process.cpdist(left_data, right_data, scorer=self.func, workers=self.workers)
    
    def _match_square(self, left_data: List[str], right_data: List[str]):
        return process.cdist(left_data, right_data, scorer=self.func, workers=self.workers)
    
    def _func_from_str(self, func_str: str):
        """Get callable from name string.
        Args:
            func_str (str): Name of the function
        Returns:
            Callable[[str, str], float]: calleble string matching function
        """
        return eval(func_str)
    
class SimpleMatcher(Matcher):
    """Implements a simple Matcher.
    Args:
        func (Callable[[str, str], float]): matching function.
    """
    def __init__(self, func: Callable[[str, str], float]) -> None:
        self.func = func

    def _match_pairwise(self, left_data: List[str], right_data: List[str]):
        return pd.DataFrame(zip(left_data, right_data), columns=["left", "right"]).apply(
            lambda x: self.func(x.left, x.right), axis=1).values
    
    def _match_square(self, left_data: List[str], right_data: List[str]):
        n1, n2 = len(left_data), len(right_data)
        m = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                m[i][j] = self.func(left_data[i], right_data[j])
        return m
