import argparse
import pandas as pd
from typing import List 
from tqdm import tqdm
from Utils import (SONG_ATTRS, simplify_string, make_taglist, retag_matches)
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

def main():

    args = parse_args()

    data = pd.read_parquet(args.input)

    exact = args.min_r == None
    if exact:
        min_r = 100
    else:
        min_r = args.min_r
    print(f"Creating dataset: ALL={args.all}; min_r={min_r}")

    def get_second_level_columns(data: pd.DataFrame, first_name: str) -> List[str]:
        return data.columns.get_level_values(1)[data.columns.get_level_values(0) == first_name]

    tqdm.pandas()

    song_attrs = [attr for attr in SONG_ATTRS if attr in get_second_level_columns(data, "shs_processed")]
    
    # 1. IOB tag lists
    yt_processed = "yt_processed"
    for yt_attr in get_second_level_columns(data, yt_processed):
        print(f"Processing text from attr: {yt_attr}")
        data[("TEXT", yt_attr)] = data[(yt_processed, yt_attr)].progress_apply(lambda x: simplify_string(x).split())
        series = data.progress_apply(make_taglist, args=(song_attrs, args.baseline_names, args.all, min_r, (yt_processed, yt_attr)), axis=1)
        data[("IOB", yt_attr)] = series.apply(lambda x: x[0])
        data[("IOB_Indel", yt_attr)] = series.apply(lambda x: x[1])
        print("Retag IOBs...")
        data[("IOB", yt_attr)] = data.progress_apply(lambda x: retag_matches(x[("TEXT", yt_attr)], x[("IOB", yt_attr)]), axis=1)

    # 2. stack
    data = data[""].join(
        data[["TEXT", "IOB", "IOB_Indel"]].stack(future_stack=True).reset_index(level=-1).rename(
            columns={'level_1': 'Attr'})
            )
    
    # 2. attach partitions
    data = attach_segment(data, "part")

    data.to_parquet(args.output)

    # print statistics
    data["id"] = data["yt_id"] + '_' + data["Attr"] 
    print(
        data[["Attr", "part", "id"]].groupby(
        ["part", "Attr"], as_index=False
                                            ).count().pivot_table(
                        values="id", columns="Attr", index="part"
                                                                )
        )

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