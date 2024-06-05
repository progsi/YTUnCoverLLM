import argparse
import os
import pandas as pd
import numpy as np
from Utils import SONG_ATTRS
from typing import List


def write_biotag(data: pd.DataFrame, filepath: str, IOB_col : str):
    """Writes a dataframe to NER IOB tag format. From:
    https://stackoverflow.com/questions/67200114/convert-csv-data-into-conll-bio-format-for-ner
    Args:
        data (pd.DataFrame): the dataframe.
        filepath (str): the output filepath.
        IOB_col (str): name of IOB column
    """
    with open(filepath, "w") as f_out:
        for _, line in data.iterrows():
            for txt, tag in zip(line["TEXT"], line[IOB_col]):
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


def __find_sublist_indices(superlist: np.ndarray, sublist: np.ndarray) -> List[int]:
    """
    Find all indices where the sublist occurs in the array of text_list.

    Args:
        text_list (np.ndarray): The array of words to search within.
        sublist (np.ndarray): The sublist of words to search for.

    Returns:
        List[int]: A list of starting indices where the sublist occurs in the array of words.
    """
    indices = []
    sublist_len = len(sublist)
    
    for i in range(len(superlist) - sublist_len + 1):
        if np.array_equal(superlist[i:i + sublist_len], sublist):
            indices.append(i)
    
    return indices

def __retag_matches(text_list: np.ndarray, tag_list: np.ndarray) -> np.ndarray:
    """
    Retag the tag_list array with the correct IOB tags based on all matching entities.

    Args:
        text_list (np.ndarray): The array of text_list.
        tag_list (np.ndarray): The array of IOB tags.

    Returns:
        np.ndarray: The updated array of IOB tags.
    """

    entities = []
    i = 0
    while i < len(text_list):
        if tag_list[i].startswith('B-'):
            entity_words = []
            entity_tags = []
            while i < len(text_list) and (tag_list[i].startswith('I-') or (tag_list[i].startswith('B-') and not entity_words)):
                entity_words.append(text_list[i])
                entity_tags.append(tag_list[i])
                i += 1
            entities.append((np.array(entity_words), np.array(entity_tags)))
        else:
            i += 1
    
    for entity_words, entity_tags in entities:
        indices = __find_sublist_indices(text_list, entity_words)
        for index in indices:
            tag_list[index:index + len(entity_words)] = entity_tags
    
    return tag_list


def main():

    args = parse_args()
    assert args.minimum_ents >= 0, "Parameter --minimum_ents cannot be negative!" 

    data = pd.read_parquet(args.input)

    # only retain where attributes are there after processing
    data = __drop_with_missing_attrs(data, SONG_ATTRS)

    # manually retag match-based for partial
    data[args.IOB_col] = data.apply(lambda x: __retag_matches(x.TEXT, x.IOB_PARTIAL), axis=1)

    # only retain samples with minimum number of entity labels
    data = data[data[args.IOB_col].apply(lambda x: len(set([e.replace("B-", "").replace("I-", "") for e in x]))) >=  args.minimum_ents]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.ignore_split:
        # if split is ignored, only test set is written.
        out_path = os.path.join(args.output, "test.bio")
        write_biotag(data, out_path, )
    else:
        for split in ["TRAIN", "TEST", "VALIDATION"]:
            out_path = os.path.join(args.output, split.lower() + ".bio")
            data_out = data.loc[data["split"].apply(lambda x: x in split)]
            # write only if contains anything
            if len(data_out) > 0:
                write_biotag(data_out, out_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Format parquet dataset with IOB tag lists to IOB format.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output IOB files.')
    parser.add_argument('-m', '--minimum_ents', type=int, default=2, help='Number of non-null entity labels required. Defaults to 2 since we mostly expect to have the title and performer.')
    parser.add_argument('-c', '--IOB_col', type=str, default="IOB_PARTIAL", help='Name of the IOB column.')
    parser.add_argument('--ignore_split', action='store_true', help='Whether to ignore the default split given in the column named "split".')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()