import argparse
import os
import pandas as pd
from Utils import SONG_ATTRS, find_sublist_indices
from typing import List
from tqdm import tqdm


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

def append_write_path(data: pd.DataFrame) -> pd.DataFrame:
    """Get the subdir name for dataset.
    Args:
        data (pd.DataFrame): dataframe without write path col
    Returns:
        pd.DataFrame: dataframe with write path col
    """
    def __get_write_path(part: str) -> str:
        if part in ["both_100", "medium"]:
            return "complete"
        elif part in ["Artist_nan", "both_nan", "WoA_nan"]:
            return os.path.join("incomplete", part)
        
    data["write_path"] = data.part.apply(__get_write_path)
    return data

def main():

    args = parse_args()

    data = pd.read_parquet(args.input)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    data = append_write_path(data)

    for write_path in data.write_path.unique():
        data_part = data.loc[data.write_path == write_path]

        output_dir = os.path.join(args.output, write_path)
        os.makedirs(output_dir, exist_ok=True)

        if args.ignore_split:
            # if split is ignored, only test set is written.
            out_path = os.path.join(output_dir, "test.IOB")
            write_biotag(data_part, out_path, "IOB")
        else:
            for split in ["TRAIN", "TEST", "VALIDATION"]:
                out_path = os.path.join(output_dir, split.lower() + ".IOB")
                data_out = data_part.loc[data_part["split"].apply(lambda x: x in split)]
                # write only if contains anything
                if len(data_out) > 0:
                    write_biotag(data_out, out_path, "IOB")

def parse_args():
    parser = argparse.ArgumentParser(description='Format parquet dataset with IOB tag lists to IOB format.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output IOB files.')
    parser.add_argument('--ignore_split', action='store_true', help='Whether to ignore the default split given in the column named "split".')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()