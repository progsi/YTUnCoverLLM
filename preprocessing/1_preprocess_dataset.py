import argparse
import pandas as pd
import numpy as np
import importlib  
import re
import sys
from Utils import SONG_ATTRS, YT_ATTRS, replace_linebreaks_tabs, basic_preprocessing
from Processor import PerformerStringPreprocessor, TitleStringPreprocessor
from typing import List, Callable
sys.path.append('baseline')
preprocessing = importlib.import_module("music-ner-eacl2023.music-ner.datasets.preprocessing")


def __apply_preprocessing(data: pd.DataFrame, processing_pipeline: Callable) -> pd.DataFrame:
    """Apply preprocessing.
    Args:
        data (pd.DataFrame): the dataframe
        processing_pipeline (Callable): the preprocessing function
    Returns:
        pd.DataFrame: processed data
    """
    # process concatenated YT attribute values
    data["yt_processed"] = processing_pipeline(
        data.apply(lambda x: '. '.join([x[attr] for attr in YT_ATTRS]), axis=1).to_list()
    )
    
    # apply preprocessing from baseline paper
    for attr in SONG_ATTRS:
        if attr in data.columns:
            series = data[attr]
            if series.apply(isinstance, args=(list,)).all():
                # apply preprocessing to each perf string individually
                data[attr + '_processed'] = series.apply(lambda x: processing_pipeline(replace_linebreaks_tabs(pd.Series(x))))
            elif series.apply(isinstance, args=(str,)).all():
                # apply preprocessing to whole string
                data[attr + '_processed'] = processing_pipeline(replace_linebreaks_tabs(series))
    return data

def main():

    args = parse_args()
    data = pd.read_parquet(args.input)

    # split SHS metadata
    if args.split:
        # split artists
        performer_processor = PerformerStringPreprocessor()
        data = performer_processor(data)

        # split titles
        title_processor = TitleStringPreprocessor()
        data = title_processor(data)

    if args.baseline:
        # apply baseline processing
        processor = preprocessing.WrittenQueryProcessor()
        pipeline = processor.processing_pipeline
    else:
        # apply simpler preprocessing pipeline
        pipeline = basic_preprocessing

    data = __apply_preprocessing(data, pipeline)

    data.to_parquet(args.output)

def parse_args():
    parser = argparse.ArgumentParser(description='Transform parquet files with song and youtube metadata into processed dataframe.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output parquet file.')
    parser.add_argument('--split', action='store_true', help='Whether to split performer strings by criteria to detect multiple performers (eg. at "featuring").')
    parser.add_argument('--baseline', action='store_true', help='Whether to apply the preprocessing pipeline by the baseline for all strings.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()