import argparse
import os
import pandas as pd
import sys
from Utils import write_biotag

def main():

    args = parse_args()

    data = pd.read_parquet(args.iob_dataset)
    data_annotated = pd.read_parquet(args.human_annotation_file)

    annotator = "ANNOT5"
    cols = ["set_id", "yt_id", "IOB"]
    data_annotated = data_annotated[("IOB", annotator)].reset_index()[cols]
    data_annotated.columns = cols
    data_annotated = data_annotated.rename(columns={"IOB": "IOB_annotated"})

    data = pd.merge(data, data_annotated, on=["set_id", "yt_id"], how="left")

    data.loc[~data.IOB_annotated.isna(), "IOB"] = data.loc[~data.IOB_annotated.isna(), "IOB_annotated"]
    
    # filtering
    data = data.loc[~data.IOB_annotated.isna() | data.part.isin(["both_100", "medium"])]
    # TODO: consider eventually using different attributes
    data = data.loc[data.Attr == "video_title"]
    
    data.to_parquet(args.output_dir + "data.parquet")

    # write IOB files
    for split in ["TRAIN", "TEST", "VALIDATION"]:
        out_path = os.path.join(args.output_dir, split.lower() + ".IOB")
        data_out = data.loc[data["split"].apply(lambda x: x in split)]
        write_biotag(data_out, out_path, "IOB")


def parse_args():
    parser = argparse.ArgumentParser(description='Format parquet dataset with IOB tag lists to IOB format.')
    parser.add_argument('--human_annotation_file', type=str, help='Path with human annotations (output from ner_validator).')
    parser.add_argument('-i', '--iob_dataset', type=str, help='IOB dataset parquet file.')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory.')
    parser.add_argument('-a', '--annotator', type=str, help='Which annotator to trust.', default="ANNOT5")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()