import argparse
import pandas as pd
import os
from src.preprocessing.Processor import PerformerStringPreprocessor
from src.preprocessing.Utils import (unicode_normalize, remove_brackets_and_all_content, 
                   remove_bracket_only, replace_linebreaks_tabs)

PROCESSING_ATTRS = ["artist_original", "artist_perf", "composer"]
PREPROCESSOR = PerformerStringPreprocessor()


def artist_partly_correct(row: pd.Series, answer_col: str):
    """Checking if partly correct ("Related")
    Args:
        row (pd.Series): series of answers
        answer_col (str): col of answer
    Returns:
        bool: partial correctness
    """
    if not pd.isna(row[answer_col]) and isinstance(row.performer, list):
        artists = row[answer_col].split()
        if row.artist_original:
            artists += row.artist_original.split()
        if row.composer:
            artists += row.composer.split()
        for artist in artists:
            for performer in row.performer:
                if artist in performer:
                    return True
    else: 
        return None
    return False

def AW1_correct(row: pd.Series) -> bool:
    """Correctness check for memorization test.
    Args:
        row (pd.Series): series of answers
    Returns:
        bool: Correctness
    """
    if not pd.isna(row.AW1) and not pd.isna(row.artist_original):
        aws = row.AW1.split()
        return any([aw.lower() in [a.lower() for a in row.artist_original.split()] for aw in aws])
    return

def AW2_correct(row: pd.Series) -> bool:
    """Correctness check for memorization test.
    Args:
        row (pd.Series): series of answers
    Returns:
        bool: Correctness
    """
    if not pd.isna(row.AW2) and not pd.isna(row.composer):
        aws = row.AW2.split()
        return any([aw.lower() in [c.lower() for c in row.composer.split()] for aw in aws])
    return

def preprocessing(s: str) -> str:
    """Preprocess string.
    Args:
        s (str): attribute value

    Returns:
        str: processed attribute
    """
    s = unicode_normalize(s)
    # remove brackets with one-word content eg "[us]"
    s = remove_brackets_and_all_content(s)
    # remove brackets but keep content, eg. when (feat. Metallica) keep feat. Metallica
    s = remove_bracket_only(s)
    # split performers by defined separators
    l = PREPROCESSOR.split_performers(replace_linebreaks_tabs(s))
    # also consider performer names without artists
    l = PREPROCESSOR.article_preprocessing(l)
    return ','.join(l)
    
def consolidate_tests(input_dir: str, df_metadata: str) -> pd.DataFrame:
    """Consolidate test results from memorization tests per model
    Args:
        input_dir (str): filepath to output of memorization tests per model
        metadata (str): metadata file
    Returns:
        pd.DataFrame: consolidated test results
    """
    dfs = []
    for fn in os.listdir(input_dir):
        fp = os.path.join(input_dir, fn)
        df = pd.read_json(fp, lines=True, orient="records")
        if "set_id" in df.columns:
            merged_data = pd.merge(df, df_metadata, how="left", on="set_id")
            merged_data["filename"] = fn
            dfs.append(merged_data)
    return pd.concat(dfs)

import pandas as pd

def load_and_process_data(path: str) -> pd.DataFrame:
    """
    Load data from a parquet file, process it, and return the DataFrame.
    Args:
        path (str): The path to the parquet file.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    data = pd.read_parquet(path).drop_duplicates(subset="id")
    data["yt_id"] = data.id.apply(lambda x: x.split("_", 1)[1])

    data["has_WoA"] = data.IOB.apply(lambda x: "B-WoA" in x)
    data["has_Artist"] = data.IOB.apply(lambda x: "B-Artist" in x)
    data["index"] = data.groupby("subset").cumcount()
    return data

def aggregate_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """Correctness Booleans to strings.
    Args:
        df (pd.DataFrame): 
    Returns:
        pd.DataFrame: 
    """
    model_list = []

    for col in df.columns:
        model = col[0]
        if model != '' and model not in model_list:
            model_list.append(model)

            df_model = df[model]

            aw_cols = [c for c in df_model.columns if c.startswith("AW")]
            aw_cols_correct = [c for c in aw_cols if "Correct" in c]

            def correctness(row):
                if row[aw_cols].T.sum() == 0:
                    return "None"
                elif row[aw_cols_correct].T.sum() >= 2:
                    return "Correct"
                else:
                    return "Partial"

            df[(model, "Correctness")] = df_model.apply(correctness, axis=1)
            df[(model, "Correctness")].value_counts()
    
    df = df.reset_index()
    df.columns = ['_'.join(col).rstrip("_") for col in df.columns]
    return df

def filestr2modelstr(filename: str) -> str:
    """prettify the name of files/models.
    Args:
        filename (str): 
    Returns:
        str: prettified name
    """
    name_without_extension = os.path.splitext(filename)[0]
    capitalized_name = ' '.join(word.capitalize() for word in name_without_extension.split('_'))
    return capitalized_name

def pivot(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ['set_id', 'work_id', 'perf_id', 'Model']
    aw_cols = [col for col in df.columns if col.startswith('AW') and ":" in col]

    df[id_cols + aw_cols]
    df_melted = df.melt(id_vars=id_cols, value_vars=aw_cols)

    return df_melted.pivot_table(
        index=['set_id', 'work_id', 'perf_id'],
        columns=['Model', 'variable'],
        values='value',
        aggfunc='first'  # 'first' because we assume there is no aggregation needed if values are unique
    )
    

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process some directories and metadata.")
    parser.add_argument('--input_dir', type=str, default="output/memorization/", help='Input directory with jsonl per LLM.')
    parser.add_argument('--datasetfile', type=str, default="data/dataset/reddit+shsyt/data.parquet", help='Filepath to parquet file with dataset')
    parser.add_argument('--metadatafile', type=str, default="data/source/shs100k2.jsonl", help='Metadata filepath (jsonl)')
    parser.add_argument('--output', type=str, default="output/memorization/memorization2.jsonl", help='Output filepath')
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_dir = args.input_dir
    dataset = args.datasetfile
    metadata = args.metadatafile
    
    df_metadata = pd.read_json(metadata, lines=True, orient='records')
    
    # get memorization test results per LLM
    df = consolidate_tests(input_dir, df_metadata)
    df["Model"] = df.filename.apply(filestr2modelstr)
    
    # preprocess attributes
    df.artist_original = df.artist_original.apply(lambda x: preprocessing(x) if type(x) == str else None)
    df.artist_perf = df.artist_perf.apply(lambda x: preprocessing(x) if type(x) == str else None)
    df.composer = df.composer.apply(lambda x: preprocessing(x) if type(x) == str else None)
    
    # determine correctness
    df["AW1: Correct"] = df.apply(AW1_correct, axis=1)
    df["AW1: Related"] = df.apply(lambda x: artist_partly_correct(x, "AW1"), axis=1)
    df["AW2: Correct"] = df.apply(AW2_correct, axis=1)
    df["AW2: Related"] = df.apply(lambda x: artist_partly_correct(x, "AW2"), axis=1)

    df = pivot(df)
    df = aggregate_correctness(df)
    
    df = pd.merge(
        load_and_process_data(dataset), 
        df, on="set_id", how="left")
    df.to_json(args.output, lines=True, orient="records")
    
    

if __name__ == "__main__":
    main()