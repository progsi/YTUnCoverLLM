import argparse
import pandas as pd
import importlib  
import sys
from Attributes import SONG_ATTRS, CLASS_ATTRS, YT_ATTRS
from typing import List, Callable
sys.path.append('baseline')
preprocessing = importlib.import_module("music-ner-eacl2023.music-ner.datasets.preprocessing")


def __split_performers_series(performers: pd.Series, featuring_token: str = "featuring"):
    """Splits the raw performer string by handcrafted criteria.
    Args:
        performers (pd.Series): Series of lists where each element in the list is a single performer.
    """
    AND_VARIATIONS_LONG = [
    # and + genetive
    "and his", "and her", "y su", "e la sua", 
    "e la seu", "e seu", "e sua", "und sein", "und ihr", 
    "und seine", "und ihre", "et le", "et son", "et ses", 
    "et les", 
    # with...
    "with her", "with his", "with the", 
    "mit ihrem", "mit ihren", "mit seinem", "mit seinen",
    "com o seu", "com o"]
    AND_VARIATIONS_SHORT = ["&", "and", "y", "e", "et", "und", ",", "-", "con", "avec", "mit", "com"]

    # lowercase
    performers = performers.str.lower()
    # normalize punctiation
    performers = performers.str.replace(" feat. ", " feat ").str.replace(" ft. ", " ft ")
    # normalize featuring abbrv.
    performers = performers.str.replace(" feat ", f" {featuring_token} ").str.replace(" ft ", f" {featuring_token} ")

    performers = performers.str.replace(", ", " , ")

    # replace long variations
    for and_var in AND_VARIATIONS_LONG:
        performers = performers.str.replace(f" {and_var} ", f" {featuring_token} ")

    # replace short variations
    for and_var in AND_VARIATIONS_SHORT:
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

def __split_performers(data: pd.DataFrame, song_attrs: List[str]):
    """Given the data, the performer strings are split and column names are documented in a list.
    Args:
        data (pd.DataFrame): dataframe
        song_attrs (List[str]): song attribute names
    Returns:
        pd.DataFrame, List[str]: dataframe with split performers and list with col names of split columns
    """
    for attr in song_attrs:
        if attr in data.columns and "performer" in attr:
            data[attr] = __split_performers_series(data[attr].replace('\n', ' ').replace('\t', ' '))
    return data

def __apply_preprocessing(data: pd.DataFrame, processing_cols: List[str], processing_pipeline: Callable):
    """_summary_
    Args:
        data (pd.DataFrame): the dataframe
        processing_cols (List[str]): which columns to preprocess
        processing_pipeline (Callable): the preprocessing function
    Returns:
        pd.DataFrame: processed data
    """
    # process concatenated YT attribute values
    data["yt_processed"] = processing_pipeline(
        data.apply(lambda x: '. '.join([x[attr] for attr in YT_ATTRS]), axis=1))

    # apply preprocessing from baseline paper
    for attr in processing_cols:
        if attr in data.columns:
            series = data[attr]
            if series.apply(isinstance, args=(list,)).all():
                # apply preprocessing to each perf string individually
                data[attr + '_processed'] = series.apply(lambda x: processing_pipeline(pd.Series(x).replace('\n', ' ').replace('\t', ' ')))
            elif series.apply(isinstance, args=(str,)).all():
                # apply preprocessing to whole string
                data[attr + '_processed'] = processing_pipeline(series.replace('\n', ' ').replace('\t', ' '))
    return data

def main():

    args = parse_args()
    data = pd.read_parquet(args.input)

    processor = preprocessing.WrittenQueryProcessor()
    pipeline = processor.processing_pipeline

    # split performers
    if args.split:
        data = __split_performers(data, SONG_ATTRS)

    data = __apply_preprocessing(data, SONG_ATTRS, pipeline)
    # filter columns
    rel_cols = [col for col in data.columns if col in CLASS_ATTRS or '_processed' in col]

    data[rel_cols].to_parquet(args.output)

def parse_args():
    parser = argparse.ArgumentParser(description='Transform parquet files with song and youtube metadata into processed dataframe.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output parquet file.')
    parser.add_argument('--split', action='store_true', help='Whether to split performer strings by criteria to detect multiple performers (eg. at "featuring").')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()