import pandas as pd
from typing import List
import numpy as np
from Utils import (replace_linebreaks_tabs, 
                   unicode_normalize, remove_bracket_only, 
                   remove_bracket_with_one_content, remove_brackets_and_all_content)
import re
import unicodedata


class TitleStringPreprocessor:
    """Class to preprocess title strings (eg. splitting first and second titles).
    """
    def __call__(self, data: pd.DataFrame, processing_attrs: List[str] = ["title_perf", "title_work", "title"]) -> pd.DataFrame:
            """Given the data, the title processing is applied.
            Args:
                data (pd.DataFrame): dataframe
                processing_attrs (List[str]): List of attr names of columns to process.
            Returns:
                pd.DataFrame, List[str]: dataframe with split performers and list with col names of split columns
            """
            for attr in processing_attrs:
                if attr in data.columns:
                    # normalize font encodings
                    data[attr] = self.__preprocessing_pipe(data[attr])

            return data
    
    def __preprocessing_pipe(self, series: pd.Series) -> pd.Series:

        # normalize unicode fonts
        series = series.apply(unicode_normalize) 
        # split by / separator
        series = series.str.split("/")
        # add short titles, eg. remove "(acoustic)" etc.
        series = series.apply(self.__add_short_title)

        return series

    @staticmethod
    def __add_short_title(l: List[str]) -> List[str]:
        """If there is a short version of a title, add it.
        Args:
            l (List[str]): titles
        Returns:
            List[str]: list of title(s)
        """
        for s in l:
            s = s.strip()
            short_s = remove_brackets_and_all_content(s).strip()
            if short_s != s and len(short_s) > 0:
                l.append(short_s)
        return l

class PerformerStringPreprocessor:
    """Class to process performer strings (eg. splitting)
    """
    def __init__(self):
        self.articles = [
            # french
            "le", "les", 
            # spanish
            "la", "el", "los", "las", 
            # german
            "der", "die", "das", 
            # portuguese
            "a", "o", "os", "as"
            # italian
            "il", "lo", "l'", "i", "gli", "le",
            # english
            "the"]
        self.and_variations_long = [
            # and + genetive
            "and his", "and her", "y su", "e la sua", 
            "e la seu", "e seu", "e sua", "und sein", "und ihr", 
            "und seine", "und ihre", "et le", "et son", "et ses", 
            "et les", 
            # with...
            "with her", "with his", "with the", 
            "mit ihrem", "mit ihren", "mit seinem", "mit seinen",
            "com o seu", "com o"]
        self.and_variations_short = ["&", "and", "y", "e", "et", "und", "/",
                                     ",", "-", "con", "avec", "mit", "com", "with"]

    def __call__(self, data: pd.DataFrame, processing_attrs: List[str] = ["performer_perf", "performer_work", "performer"]) -> pd.DataFrame:
        """Given the data, the performer strings are split and column names are documented in a list.
        Args:
            data (pd.DataFrame): dataframe
            processing_attrs (List[str]): List of attr names of columns to process.
        Returns:
            pd.DataFrame: dataframe with split performers
        """
        for attr in processing_attrs:
            if attr in data.columns:
                data[attr] = self.__preprocessing_pipe(data[attr])
        return data

    def __preprocessing_pipe(self, series: pd.Series) -> pd.Series:
        # normalize font encodings
        series = series.apply(unicode_normalize)
        # remove brackets with one-word content eg "[us]"
        series = series.apply(remove_bracket_with_one_content)
        # remove brackets but keep content, eg. when (feat. Metallica) keep feat. Metallica
        series = series.apply(remove_bracket_only)
        # split performers by defined separators
        series = self.__split_performers(replace_linebreaks_tabs(series))
        # also consider performer names without artists
        series = series.apply(self.__article_preprocessing)

        return series

    def __article_preprocessing(self, arr: np.array) -> List[str]:
        """For each string with an article from the pre-fixed list of articles, 
        also consider the string without the article for more robustness.
        Args:
            arr (np.array): list-like in the dataframe
        Returns:
            List[str]: initial list + strings without articles
        """
        cleaned = []
        for item in arr:
            for article in self.articles:
                article_space = article + " "
                if item.find(article_space) == 0:
                    cleaned_item = item.replace(article_space, "")
                    if cleaned_item not in cleaned:
                        cleaned.append(cleaned_item)
        return list(arr) + cleaned

    def __split_performers(self, performers: pd.Series, featuring_token: str = "featuring") -> pd.Series:
        """Splits the raw performer string by handcrafted criteria.
        Args:
            performers (pd.Series): Series of lists where each element in the list is a single performer.
        Returns:
            pd.Series:
        """
        # lowercase
        performers = performers.str.lower()
        # normalize punctiation
        performers = performers.str.replace(" feat. ", " feat ").str.replace(" ft. ", " ft ")
        # normalize featuring abbrv.
        performers = performers.str.replace(" feat ", f" {featuring_token} ").str.replace(" ft ", f" {featuring_token} ")

        performers = performers.str.replace(", ", " , ")

        # replace long variations
        for and_var in self.and_variations_long:
            performers = performers.str.replace(f" {and_var} ", f" {featuring_token} ")

        # replace short variations
        for and_var in self.and_variations_short:
            performers = performers.str.replace(f" {and_var} ", f" {featuring_token} ")

        def only_space(s):
            """Check if string contains only spaces, tabs, newlines.
            Args:
                s (str): input string
            Returns:
                bool: 
            """
            return all(char.isspace() for char in s)
        
        return performers.apply(lambda x: [t.strip() for t in x.split("featuring") if not only_space(t)])
