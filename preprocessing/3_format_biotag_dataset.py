import argparse
import os
import pandas as pd
import numpy as np
from Utils import SONG_ATTRS
from typing import List


def write_biotag(data: pd.DataFrame, filepath: str):
    """Writes a dataframe to NER BIO tag format. From:
    https://stackoverflow.com/questions/67200114/convert-csv-data-into-conll-bio-format-for-ner
    Args:
        data (pd.DataFrame): the dataframe.
        filepath (str): the output filepath.
    """
    with open(filepath, "w") as f_out:
        for _, line in data.iterrows():
            for txt, tag in zip(line["TEXT"], line["NER_TAGS"]):
                print("{}\t{}".format(txt, tag), file=f_out)
            print(file=f_out)

def __drop_with_missing_attrs(data: pd.DataFrame, attrs: List[str]):
    """For given dataframe, remove values with missing

    Args:
        data (pd.DataFrame): input dataframe
        attrs (List[str]): attribute names
    Returns:
        pd.DataFrame: filtered dataframe
    """
    def __is_missing(value: object):
        if type(value) == str:
            return value == ""
        elif type(value) == list:
            return (len(value) == 0) or not (any(e not in ['', None] for e in value))
        else:
            return value is None

    for attr in attrs:
        col = attr + '_processed'
        if col in data.columns:
            data = data.dropna(subset=[col])
            data = data.loc[~data[col].apply(__is_missing)]
    return data

def main():

    args = parse_args()
    assert args.minimum_ents >= 0, "Parameter --minimum_ents cannot be negative!" 

    data = pd.read_parquet(args.input)

    # only retain where attributes are there after processing
    data = __drop_with_missing_attrs(data, SONG_ATTRS)

    # only retain samples with minimum number of entity labels
    data = data[data.NER_TAGS.apply(lambda x: len(set([e.replace("B-", "").replace("I-", "") for e in x]))) >=  args.minimum_ents]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.ignore_split:
        # if split is ignored, only test set is written.
        out_path = os.path.join(args.output, "test.bio")
        write_biotag(data, out_path)
    else:
        for split in ["TRAIN", "TEST", "VALIDATION"]:
            out_path = os.path.join(args.output, split.lower() + ".bio")
            data_out = data.loc[data["split"].apply(lambda x: x in split)]
            # write only if contains anything
            if len(data_out) > 0:
                write_biotag(data_out, out_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Format parquet dataset with BIO tag lists to BIO format.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output BIO files.')
    parser.add_argument('-m', '--minimum_ents', type=int, default=2, help='Number of non-null entity labels required. Defaults to 2 since we mostly expect to have the title and performer.')
    parser.add_argument('--ignore_split', action='store_true', help='Whether to ignore the default split given in the column named "split".')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()