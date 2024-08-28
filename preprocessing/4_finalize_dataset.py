import argparse
import os
import pandas as pd


def get_curated_data(curated_path: str) -> pd.DataFrame:
    """Gets the curated data, based on paths to curated datasets.
    Args:
        curated_path (str): path to csv file (tab-sep) with curations.
    Returns:
        pd.DataFrame: curated data for full annotated set
    """
    data = pd.read_csv(curated_path, sep="\t")
    
    def get_correct(row):
        x = row.Curated
        if x == 1 or x == "1":
            return row.a1
        elif x == 2 or x == "2":
            return row.a2
        else:
            return row.Curated

    data.Curated = data.apply(get_correct, axis=1)
    return data

def get_human_annotations(path_annotated: str) -> pd.DataFrame:

    data = pd.read_parquet(os.path.join(path_annotated, "data.parquet"))
    data_curated = get_curated_data(os.path.join(path_annotated, "curated.csv"))

    # filter for samples with min. 2 annotations
    annotators = data.columns.get_level_values(1).unique().to_list()
    not_null_counts = data[[('IOB', annotator) for annotator in annotators]].notnull().sum(axis=1)
    data = data[not_null_counts >= 2].reset_index()

    def get_final(row):
        set_id, yt_id = row.set_id.item(), row.yt_id.item()
        mask = (data_curated.set_id == set_id) & (data_curated.yt_id == yt_id)
        curated = data_curated.loc[mask, "Curated"]
        if not curated.empty:
            iob = curated.item()
        else:
            iob = row[("IOB", annotators[-1])]
        return iob

    data[("IOB", "final")] = data.apply(get_final, axis=1)
    data = data[[("set_id", ""), ("yt_id", ""), ("IOB", "final")]]
    data.columns = data.columns.droplevel(1)
    return data.set_index(["set_id", "yt_id"])


def main():
    args = parse_args()

    data_human = get_human_annotations(args.path_annotated)
    map_human = data_human["IOB"].to_dict()

    data = pd.read_parquet("data/intermediate/shs100k2_IOB.parquet")
    data = data.loc[data.Attr == "video_title"]
    data["IOB_auto"] = data["IOB"]
    data["IOB_human"] = data.apply(lambda row: map_human.get((row.set_id, row.yt_id)), axis=1)
    data["IOB"] = data['IOB_human'].combine_first(data['IOB_auto'])

    data.to_parquet(args.output_dir + "data.parquet")

    # write IOB files
    for split in ["TRAIN", "TEST", "VALIDATION"]:
        out_path = os.path.join(args.output_dir, split.lower() + ".IOB")
        data_out = data.loc[data["split"].apply(lambda x: x in split)]
        write_biotag(data_out, out_path, "IOB")

def parse_args():
    parser = argparse.ArgumentParser(description='Format parquet dataset with IOB tag lists to IOB format.')
    parser.add_argument('-i', '--path_iob_dataset', type=str, help='IOB dataset parquet file.')
    parser.add_argument('-a', '--path_annotated', type=str, default="data/annotated/")
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()