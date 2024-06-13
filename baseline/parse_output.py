import argparse
import numpy as np
import pandas as pd
import os
from typing import List

DEFAULT_BASE_PATH = "../baseline/music-ner-eacl2023/output/shs100k2/complete"

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

def get_predict_results_path(base_path: str, model: str, filename: str = "predict_results.json") -> str:
    return os.path.join(base_path, model, filename)

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

def get_results_table(models: List[str], base_path: str = DEFAULT_BASE_PATH) -> pd.DataFrame:
    """Get results table per entity for list of all models
    Args:
        models (List[str]): list of model strings
        base_path (str, optional): Base path with result jsons. Defaults to DEFAULT_BASE_PATH.
    Returns:
        pd.DataFrame: results in table
    """
    data = pd.DataFrame()

    for model in models:
        data_model = get_results(get_predict_results_path(base_path, model))[["f1", "precision", "recall"]]
        data_model["Model"] = model
        data_model = data_model.set_index("Model", append=True)
        data = pd.concat([data, data_model], axis=0)

    return data.rename_axis(index=['Attribute', 'Scenario', 'Model']).pivot_table(
        index="Model", columns=["Attribute", "Scenario"], values=["f1", "precision", "recall"])

def get_results_overall_table(models: List[str], base_path: str = DEFAULT_BASE_PATH) -> pd.DataFrame:
    """Get overall results table.
    Args:
        models (List[str]): model strings
        base_path (str, optional): Base path with result jsons. Defaults to DEFAULT_BASE_PATH.
    Returns:
        pd.DataFrame: results table
    """
    data = pd.DataFrame()

    for model in models:
        data_model = get_overall_results(get_predict_results_path(base_path, model))
        data_model["Model"] = model
        data_model = data_model.set_index("Model", append=True)
        data = pd.concat([data, data_model], axis=0)

    return data.rename_axis(
        index=['_', 'Scenario', 'Metric', 'Model']).reset_index(drop=True, level="_").pivot_table(
            index="Model", columns=["Scenario", "Metric"], values=["macro", "micro"])

def parse_preds_baseline(model: str, base_path: str = DEFAULT_BASE_PATH) -> List[np.array]:
    """Parse predictions text file as list of lists with IOB tags.
    Args:
        model (str): model string
    Returns:
        List[np.array]: list of lists with IOB tags
    """
    path = get_predict_results_path(base_path, model, "predictions.txt")

    return parse_pred_file(path)

def parse_pred_file(path: str) -> List[np.array]:
    """Parse predictions textfile
    Args:
        path (str): path to file
    Returns:
        List[np.array]: parsed results
    """
    with open(path, "r") as f:
        content = f.read()
    return [np.array(s.split()) for s in content.split("\n") if len(s) > 0]
    
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
