import argparse
import pandas as pd
import importlib  
import sys
from Utils import SONG_ATTRS, CLASS_ATTRS, YT_ATTRS, split_performers_series
from typing import List, Callable
sys.path.append('baseline')
preprocessing = importlib.import_module("music-ner-eacl2023.music-ner.datasets.preprocessing")


def __split_performers(data: pd.DataFrame, song_attrs: List[str]) -> pd.DataFrame:
    """Given the data, the performer strings are split and column names are documented in a list.
    Args:
        data (pd.DataFrame): dataframe
        song_attrs (List[str]): song attribute names
    Returns:
        pd.DataFrame, List[str]: dataframe with split performers and list with col names of split columns
    """
    for attr in song_attrs:
        if attr in data.columns and "performer" in attr:
            data[attr] = split_performers_series(data[attr].replace('\n', ' ').replace('\t', ' '))
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

    data.to_parquet(args.output)

def parse_args():
    parser = argparse.ArgumentParser(description='Transform parquet files with song and youtube metadata into processed dataframe.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output parquet file.')
    parser.add_argument('--split', action='store_true', help='Whether to split performer strings by criteria to detect multiple performers (eg. at "featuring").')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()