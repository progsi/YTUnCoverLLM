import os
import json
import pandas as pd
import numpy as np
import torch
from typing import Tuple, List, Dict
from src.preprocessing.Utils import overlap


def read_textfile(path: str):

    with open(path, "r") as f:
        content = f.read()
    return content

def read_jsonfile(path: str):

    with open(path, "r") as f:
        content = json.load(f)
    return content

def read_jsonlines(file_path: str) -> List[str]:
    """
    Reads a JSON Lines (jsonl) file and returns a list of JSON objects.
    Parameters:
        file_path (str): The path to the JSON Lines file.
    Returns:
         List[str]: A list of JSON objects.
    """
    json_objects = []
    with open(file_path, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            json_objects.append(json_object)
    return json_objects

def write_jsonlines(file_path: str, json_objects: List[Dict]) -> None:
    """
    Writes a list of JSON objects to a JSON Lines (jsonl) file.
    
    Parameters:
        file_path (str): The path to the JSON Lines file.
        json_objects (List[Dict]): A list of JSON objects.
        
    Returns:
        None
    """
    with open(file_path, 'w') as file:
        for json_object in json_objects:
            file.write(json.dumps(json_object) + '\n')

def get_target_matrix(data: pd.DataFrame):
    """Generates the binary square matrix of cover song relationships between all
    elements in the dataset.
    Returns:
        np.array: binary square matrix
    """
    set_ids = data["set_id"].values
    target = (set_ids[:, None] == set_ids)
    return torch.from_numpy(target).to(dtype=torch.int)

def get_concat(df: pd.DataFrame, attrs: List[str]) -> List[str]:
    """Get concated list of strings from dataframe.
    Args:
        df (pd.DataFrame): input dataframe
        attrs (List[str]): which attributes to consider
    Returns:
        List[str]: concated strings in list
    """
    return df[attrs].apply(lambda row: ' '.join(map(str, row)), axis=1).to_list()

def get_left_right_concat(df: pd.DataFrame, left_attrs: List[str], right_attrs: List[str]) -> Tuple[List[str],List[str]]:
    """Helper to get left and right concatenated as list.
    Args:
        df (pd.DataFrame): DataFrame with metadata
        left_attrs (List[str]): attributes of left side
        right_attrs (List[str]): attributes of right side.
    Returns:
        List[str], List[str]: left and right data strings.
    """
    left_data = get_concat(df, left_attrs)
    right_data =  get_concat(df, right_attrs)
    
    return left_data, right_data

def get_concat_col_name(left_attrs: List[str], right_attrs: List[str]) -> Tuple[List[str], List[str]]:
    """Generate column name for the col with the pairwise similarity of concatenated columns 
    (eg. entity-level).
    Args:
        left_attrs (List[str]): attributes on the left side
        right_attrs (List[str]): attributes on the right side
    """
    return ('+'.join([attr for attr in left_attrs]), '+'.join([attr for attr in right_attrs]))  


def get_ents_spans(IOB: List[str]) -> Dict[Tuple[int, int], str]:
    """Get ent spans from a list of IOB tags.
    Args:
        IOB (List[str]): IOB tags
    Returns:
        Dict[Tuple[int, int], str]: mapping of ent spans
    """
    ents = {}
    cur_type = None
    start = None

    for i, token in enumerate(IOB):
        if token.startswith('B-'):
            if cur_type:
                ents.setdefault(cur_type, []).append((start, i - 1))
            cur_type = token[2:]
            start = i
        elif token.startswith('I-') and cur_type == token[2:]:
            continue
        else:
            if cur_type:
                ents.setdefault(cur_type, []).append((start, i - 1))
                cur_type = None

    if cur_type:
        ents.setdefault(cur_type, []).append((start, len(IOB) - 1))

    return ents

def get_missing_ents(IOB_true: List[str], IOB_pred: List[str]) -> Dict[Tuple[int, int], str]:
    """Get missing entities based on true and pred IOBs.
    Args:
        IOB_true (List[str]): 
        IOB_pred (List[str]): 
    Returns:
        Dict[Tuple[int, int], str]: mapping of missing entities by spans
    """
    ents_true, ents_pred = get_ents_spans(IOB_true), get_ents_spans(IOB_pred)

    missing_ents = {}
    for ent, spans_true in ents_true.items():
        for span_true in spans_true:
            spans_pred = ents_pred.get(ent)
            if not spans_pred or not any([overlap(span_true, span_pred) for span_pred in spans_pred]):
                missing_ents[span_true] = ent
    return missing_ents

def get_spurious_ents(IOB_true: List[str], IOB_pred: List[str]) -> Dict[Tuple[int, int], str]:
    """Get spurious entities based on true and pred IOBs.
    Args:
        IOB_true (List[str]): 
        IOB_pred (List[str]): 
    Returns:
        Dict[Tuple[int, int], str]: mapping of missing entities by spans
    """
    ents_true, ents_pred = get_ents_spans(IOB_true), get_ents_spans(IOB_pred)

    spurious_ents = {}
    for ent, spans_pred in ents_pred.items():
        for span_pred in spans_pred:
            spans_true = ents_true.get(ent)
            if not spans_true or not any([overlap(span_true, span_pred) for span_true in spans_true]):
                spurious_ents[span_pred] = ent
    return spurious_ents
         
def get_incorrect_ents(IOB_true: List[str], IOB_pred: List[str]) -> Dict[Tuple[int, int], str]:
    """Get incorrect entities based on true and pred IOBs.
    Args:
        IOB_true (List[str]): 
        IOB_pred (List[str]): 
    Returns:
        Dict[Tuple[int, int], str]: mapping of incorrect entities by spans
    """
    ents_true, ents_pred = get_ents_spans(IOB_true), get_ents_spans(IOB_pred)

    incorrect_ents = {}
    for ent, spans_true in ents_true.items():
        for span_true in spans_true:
            spans_pred = ents_pred.get(ent)
            if spans_pred:
                for span_pred in ents_pred[ent]:
                    if span_pred != span_true and overlap(span_true, span_pred): 
                        incorrect_ents[span_pred] = ent
    return incorrect_ents

def get_error_analysis(data: pd.DataFrame, pred_col: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    series_missing = data.apply(lambda row: get_missing_ents(row["IOB"], row[pred_col]), axis=1)
    series_spurious = data.apply(lambda row: get_spurious_ents(row["IOB"], row[pred_col]), axis=1)
    series_incorrect = data.apply(lambda row: get_incorrect_ents(row["IOB"], row[pred_col]), axis=1)
    return series_missing, series_spurious, series_incorrect

def read_IOB_file(path: str) -> Tuple[List[np.array], List[np.array]]:
    """Read in an IOB textfile
    Args:
        path (str): path to IOB file
    Returns:
        Tuple[List[np.array], List[np.array]]: list of texts and list of IOBs
    """
    with open(path, "r") as f:
        content = f.readlines()
    
    cur_words = []
    cur_tags = []
    words = []
    tags = []
    for row in content:
        if row == "\n":
            words.append(np.array(cur_words, dtype="object"))
            tags.append(np.array(cur_tags, dtype="object"))
            cur_words = []
            cur_tags = []
        else:
            row = row.replace("\n", "").split("\t")
            cur_words.append(row[0])
            cur_tags.append(row[1])
    return words, tags

def parse_preds(file_path: str) -> List[np.array]:
    """Parse predictions flexibally. File can be a jsonl or a textfile.
    Args:
        file_path (str): 
    Returns:
        List[np.array]: 
    """
    if file_path.endswith(".jsonl"):
        content = read_jsonlines(file_path)
        return [np.array(list(d.values()), dtype="object") for d in content]
    else:
        return parse_pred_file(file_path)


def transform_to_dict(words: np.array, tags: np.array) -> Dict[str, List[str]]:
    """Transform words and tags per word to dict format
    Args:
        words (np.array): 
        tags (np.array): 
    Returns:
        Dict[str, List[str]]: 
    """
    entities = {}
    current_entity = []
    current_label = None

    for word, tag in zip(words, tags):
        if tag == 'O':  # Outside of an entity
            if current_entity:
                entity_str = ' '.join(current_entity)
                if current_label in entities:
                    entities[current_label].append(entity_str)
                else:
                    entities[current_label] = [entity_str]
                current_entity = []
                current_label = None
        else:
            prefix, label = tag.split('-')
            if prefix == 'B':  # Beginning of a new entity
                if current_entity:
                    entity_str = ' '.join(current_entity)
                    if current_label in entities:
                        entities[current_label].append(entity_str)
                    else:
                        entities[current_label] = [entity_str]
                current_entity = [word]
                current_label = label
            elif prefix == 'I' and current_label == label:  # Inside the same entity
                current_entity.append(word)
            else:
                # Handle possible error in IOB tagging
                if current_entity:
                    entity_str = ' '.join(current_entity)
                    if current_label in entities:
                        entities[current_label].append(entity_str)
                    else:
                        entities[current_label] = [entity_str]
                current_entity = [word]
                current_label = label
    
    # Append the last entity if any
    if current_entity:
        entity_str = ' '.join(current_entity)
        if current_label in entities:
            entities[current_label].append(entity_str)
        else:
            entities[current_label] = [entity_str]

    return entities

def get_true_pred_entities(model: str, base_path: str):
    """Gets the dataframe to compare true and pred entities.
    Args:
        model (str): 
        base_path (str): 
    """
    # load data
    label_path = base_path.replace("/output/", "/data/")
    label_path = os.path.join(label_path, "test.bio")

    texts, IOB_true = read_IOB_file(label_path)
    IOB_pred = parse_preds_baseline(model, base_path)

    # collect data
    error_data = {}
    for i, (text, iob_true, iob_pred) in enumerate(zip(texts, IOB_true, IOB_pred)):
        error_data[i] = {
            "true": transform_to_dict(text, iob_true),
            "predicted": transform_to_dict(text, iob_pred)
        }

    # make df
    return pd.concat(
        [
            pd.json_normalize(pd.DataFrame(error_data).T["true"]).add_suffix("_true"),
            pd.json_normalize(pd.DataFrame(error_data).T["predicted"]).add_suffix("_pred")
        ], axis=1
    ).assign(text=[' '.join(t) for t in texts])[
        ["text", "Artist_true", "Artist_pred", "WoA_true", "WoA_pred"]
        ]

def clean_dict(d):
    """
    Recursively remove keys from a dictionary if they map to empty lists.
    """
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items() if clean_dict(v) != {} and v != []}
    elif isinstance(d, list):
        return [clean_dict(item) for item in d if clean_dict(item) != {} and item != []]
    else:
        return d




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

def read_and_prepare(input_path: str) -> pd.Series:
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
    series = read_and_prepare(input_path) 
    return __to_multiindex_dataframe(series[(series.index.map(__is_agg_col))])

def get_results(input_path: str) -> pd.DataFrame:
    """Get overall results from json as dataframe
    Args:
        input_path (str): 
    Returns:
        pd.DataFrame: 
    """
    series = read_and_prepare(input_path) 
    return __to_multiindex_dataframe(series[~(series.index.map(__is_agg_col)) & ~(series.index.map(__is_predict_col))])

def get_overview(input_path: str) -> pd.DataFrame:
    series = read_and_prepare(input_path) 
    return series[series.index.map(__is_predict_col)]

def get_results_table(models: List[str], base_path: str) -> pd.DataFrame:
    """Get results table per entity for list of all models
    Args:
        models (List[str]): list of model strings
        base_path (str, optional): Base path with result jsons..
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

def get_results_overall_table(models: List[str], base_path: str) -> pd.DataFrame:
    """Get overall results table.
    Args:
        models (List[str]): model strings
        base_path (str, optional): Base path with result jsons.
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
            index="Model", columns=["Scenario", "Metric"], values=["macro", "micro"]).T

def parse_preds_baseline(model: str, base_path: str) -> List[np.array]:
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
    


