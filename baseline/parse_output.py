import argparse
import json
import pandas as pd
import os


def __is_agg_col(colname: str) -> bool:
    return colname.endswith("_micro") or colname.endswith("_macro")

def __is_predict_col(colname: str) -> bool:
    return colname in ["loss", "runtime", "samples_per_second", "steps_per_second"]


def __to_multiindex_dataframe(series: pd.Series) -> pd.DataFrame:
    """
    Splits the Series' index into a MultiIndex and pivots the last level to columns.
    
    Args:
        data (pd.Series): The input Series with an index to be split.
        
    Returns:
        pd.DataFrame: The DataFrame with a MultiIndex and pivoted columns.
    """
    # Split index into multiindex tuples
    multiindex_tuples = [tuple(idx.split('_')) for idx in series.index]
    multiindex = pd.MultiIndex.from_tuples(multiindex_tuples)

    # Assign the new multiindex to the Series
    series.index = multiindex

    return series.unstack(level=-1)

def __read_and_prepare(input_path: str) -> pd.Series:
    """Read and prepare parsed data.
    Args:
        input_path (str): json file
    Returns:
        pd.Series: 
    """
    series = pd.read_json(input_path, typ="series")
    series.index = series.index.str.replace("ent_type", "type")
    series.index = series.index.str.replace("predict_", "")
    return series

def get_overall_results(input_path: str) -> pd.DataFrame:
    """Get overall results from json as dataframe
    Args:
        input_path (str): 
    Returns:
        pd.DataFrame: 
    """
    series = __read_and_prepare(input_path) 
    return __to_multiindex_dataframe(series[(series.index.map(__is_agg_col))])

def get_results(input_path: str) -> pd.DataFrame:
    """Get overall results from json as dataframe
    Args:
        input_path (str): 
    Returns:
        pd.DataFrame: 
    """
    series = __read_and_prepare(input_path) 
    return __to_multiindex_dataframe(series[~(series.index.map(__is_agg_col)) & ~(series.index.map(__is_predict_col))])

def get_overview(input_path: str) -> pd.DataFrame:
    series = __read_and_prepare(input_path) 
    return series[series.index.map(__is_predict_col)]

def main(input_path: str, output_dir: str):

    get_overall_results(input_path).to_csv(os.path.join(output_dir, "overall.csv"))
    get_results(input_path).to_csv(os.path.join(output_dir, "results.csv"))

    print(get_overview(input_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and process JSON data.")
    parser.add_argument('input', type=str, help='Path with input json file.')
    parser.add_argument('output', type=str, help='Path to save output csv file.')
    args = parser.parse_args()
    
    main(args.input, args.output)
