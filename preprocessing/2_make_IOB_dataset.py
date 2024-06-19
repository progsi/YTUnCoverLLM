import argparse
import pandas as pd
from typing import List 
from tqdm import tqdm
from Utils import (SONG_ATTRS, simplify_string, find_sublist_indices, make_taglist)
import numpy as np


def __get_max_indel(row: pd.Series, ent_name: str = "WoA") -> float:
    """Get maximum indel distance for matched entity.
    Args:
        row (pd.Series): row in dataframe
        ent_name (str, optional): entity name. Defaults to "WoA".
    Returns:
        float: _description_
    """
    def __get_ent_inds(l: List[str], ent_name: str = "WoA") -> List[int]:
        """Given a list, get indices with ent_name tags
        Args:
            l (List[str]): 
            ent_name (str, optional): entity name. Defaults to "WoA".
        Returns:
            List[int]: list of indices
        """
        return [i for (i, e) in enumerate(l) if ent_name in e]
    dists = [row.IOB_Indel[i] for i in __get_ent_inds(row.IOB, ent_name) if row.IOB_Indel[i] is not None]
    if len(dists) > 0:
        return max(dists)
    return None

def attach_segment(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Attach categories of difficulty to dataframe.
    Args:
        data (pd.DataFrame): 
        col_name (str): 
    Returns:
        pd.DataFrame: 
    """
    series_woa = data.apply(lambda x: __get_max_indel(x, "WoA"), axis=1)
    series_artist = data.apply(lambda x: __get_max_indel(x, "Artist"), axis=1)

    data[col_name] = None
    # both_100
    mask_both_100 = series_woa.apply(lambda x: x == 100) & series_artist.apply(lambda x: x == 100)
    data.loc[mask_both_100, col_name] = "both_100"
    # both nan
    mask_both_nan = series_artist.isna() & series_woa.isna()
    data.loc[mask_both_nan, col_name] = "both_nan"
    # woa nan
    mask_woa_nan = ~mask_both_nan & series_woa.isna()
    data.loc[mask_woa_nan, col_name] = "WoA_nan"
    # artist nan
    mask_artist_nan = ~mask_both_nan & series_artist.isna()
    data.loc[mask_artist_nan, col_name] = "Artist_nan"
    # other
    mask_medium = ~mask_both_100 & ~mask_both_nan & ~mask_woa_nan & ~mask_artist_nan
    data.loc[mask_medium, col_name] = "medium"
    return data

def retag_matches(text_list: np.ndarray, tag_list: np.ndarray) -> np.ndarray:
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
        indices = find_sublist_indices(text_list, entity_words)
        for index in indices:
            tag_list[index:index + len(entity_words)] = entity_tags
    
    return tag_list

def main():

    args = parse_args()

    data = pd.read_parquet(args.input)

    ent_names = [ent + '_processed' for ent in SONG_ATTRS if ent + '_processed' in data.columns]

    exact = args.min_r == None
    if exact:
        min_r = 100
    else:
        min_r = args.min_r
    print(f"Creating dataset: ALL={args.all}; min_r={min_r}")

    tqdm.pandas()
    data["TEXT"] = data.yt_processed.progress_apply(lambda x: simplify_string(x).split())

    # 1. generate IOB tags
    series = data.progress_apply(make_taglist, args=(ent_names, args.baseline_names, args.all, min_r), axis=1)
    data["IOB"] = series.apply(lambda x: x[0])
    data["IOB_Indel"] = series.apply(lambda x: x[1])

    # 2. attach partitions
    data = attach_segment(data, "part")

    # 3. retag for consistency
    # manually retag match-based for partial
    print("Retag IOBs...")
    data["IOB"] = data.progress_apply(lambda x: retag_matches(x.TEXT, x.IOB), axis=1)

    data.to_parquet(args.output)

def parse_args():
    parser = argparse.ArgumentParser(description='Make IOB tag list for NER task.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output parquet file.')
    parser.add_argument('-m', '--min_r', type=int, default=80, help='Minimum partial ratio to use for matching. If None or 100, exact matching is performed.')
    parser.add_argument('--baseline_names', action='store_true', help='Whether to change entity class name to the ones used in the baseline approach.')
    parser.add_argument('--all', action='store_true', help='Whether to find all or only the first occurance per entity in the string.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()