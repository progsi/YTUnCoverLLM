import argparse
import pandas as pd
import importlib  
import sys
sys.path.append('baseline')
preprocessing = importlib.import_module("music-ner-eacl2023.music-ner.datasets.preprocessing")

def main():

    # set attributes for YouTube, Song and Class Information
    YT_ATTRS = ["video_title", "channel_name", "description"]
    SONG_ATTRS = ["title", "performer", "title_perf", "performer_perf", "title_work", "performer_work"]
    CLASS_ATTRS = ["set_id", "ver_id", "yt_id"]

    args = parse_args()
    data = pd.read_parquet(args.input)

    processor = preprocessing.WrittenQueryProcessor()

    # process concatenated YT attribute values
    data["yt_processed"] = processor.processing_pipeline(
        data.apply(lambda x: '. '.join([x[attr] for attr in YT_ATTRS]), axis=1))

    # process song attributes individually
    for attr in SONG_ATTRS:
        if attr in data.columns:
            data[attr + '_processed'] = processor.processing_pipeline(data[attr].replace('\n', ' ').replace('\t', ' '))

    # filter columns
    rel_cols = [col for col in data.columns if col in CLASS_ATTRS or '_processed' in col]

    data[rel_cols].to_parquet(args.output)

def parse_args():
    parser = argparse.ArgumentParser(description='Transform parquet files with song and youtube metadata into processed dataframe.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output parquet file.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()