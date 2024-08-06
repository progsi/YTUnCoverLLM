import argparse
import os
import pandas as pd


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

def write_metadata(data: pd.DataFrame, filepath: str):
    """Writes a dataframe to metadata csv (with tab sep):
    Args:
        data (pd.DataFrame): the dataframe.
        filepath (str): the output filepath.
    """
    data = data[["set_id", "yt_id", "Attr", "title", "performer"]].rename(
        columns={"title": "WoA", "performer": "Artist"}
        )
    def join(x): return ', '.join(x)
    data["WoA"] = data["WoA"].apply(join)
    data["Artist"] = data["Artist"].apply(join)
    data.to_csv(filepath, index=None, sep="\t")


def append_write_set(data: pd.DataFrame) -> pd.DataFrame:
    """Get the subname for dataset.
    Args:
        data (pd.DataFrame): dataframe without write path col
    Returns:
        pd.DataFrame: dataframe with write path col
    """
    def __get_write_set(part: str) -> str:
        if part in ["both_100", "medium"]:
            return "complete"
        elif part in ["Artist_nan", "both_nan", "WoA_nan"]:
            return part
        
    data["write_set"] = data.part.apply(__get_write_set)
    return data

def main():

    args = parse_args()

    data = pd.read_parquet(args.input)

    # filter out rows without text
    data = data[data.TEXT.apply(len) > 0]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    data = append_write_set(data)

    for write_set in data.write_set.unique():
        data_part = data.loc[data.write_set == write_set]
        
        for attr in data_part.Attr.unique():
            
            data_attr = data_part.loc[data.Attr == attr]
            
            seed = 1
            if args.limit:
                data_attr = data_attr.sample(n=args.limit, random_state=seed)
            else:
                data_attr = data_attr.sample(frac=1, random_state=seed)

            output_dir = os.path.join(args.output, attr)
            os.makedirs(output_dir, exist_ok=True)

            if args.ignore_split:
                # if split is ignored, only test set is written.
                out_path = '-'.join((output_dir, "test.IOB"))
                write_biotag(data_part, out_path, "IOB")
                # write metadata
                write_metadata(data_part, out_path.replace(".IOB", ".metadata"))
            else:
                for split in ["TRAIN", "TEST", "VALIDATION"]:
                    out_file = '-'.join((write_set, split.lower() + ".IOB"))
                    out_path = os.path.join(output_dir, out_file)
                    data_out = data_part.loc[data_part["split"].apply(lambda x: x in split)]
                    # write only if contains anything
                    if len(data_out) > 0:
                        write_biotag(data_out, out_path, "IOB")
                        write_metadata(data_out, out_path.replace(".IOB", ".metadata"))

def parse_args():
    parser = argparse.ArgumentParser(description='Format parquet dataset with IOB tag lists to IOB format.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output IOB files.')
    parser.add_argument('--limit', type=int, help='Number of samples to write per file.', default=None)
    parser.add_argument('--ignore_split', action='store_true', help='Whether to ignore the default split given in the column named "split".')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()