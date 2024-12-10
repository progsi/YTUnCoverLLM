import argparse
import os
import re
import random
import pandas as pd
import copy
from typing import List, Tuple
from src.Utils import read_IOB_file
from src.preprocessing.Processor import PerformerStringPreprocessor
from src.preprocessing.Utils import simplify_string, remove_bracket_with_one_content, write_IOB

PERFORMER_PROCESSOR = PerformerStringPreprocessor()
def get_performer_variations(s: str) -> List[str]:
    l = PERFORMER_PROCESSOR.split_performers(s)
    l = [simplify_string(s) for s in l]
    l = PERFORMER_PROCESSOR.article_preprocessing(l)
    return l

def get_title_variations(s: str) -> List[str]:
    s = remove_bracket_with_one_content(s)
    l = s.split("/")
    l = [simplify_string(s) for s in l]
    return l


def fill_template_with_entities(template, entity_dict):
    filled_text = []
    new_iob = []
    used_entities = {key: [] for key in entity_dict.keys()}

    for token in template:
        if isinstance(token, str) and token.startswith('[') and token.endswith(']'):
            placeholder = token.strip('[]')  # Extract entity class name

            if placeholder in entity_dict:
                # Get a random, unused entity from the list for this placeholder
                available_entities = list(set(entity_dict[placeholder]) - set(used_entities[placeholder]))
                if available_entities:
                    chosen_entity = random.choice(available_entities)
                    used_entities[placeholder].append(chosen_entity)
                else:
                    # If we've exhausted all unique entities, re-use from the list
                    chosen_entity = random.choice(entity_dict[placeholder])

                # Tokenize the chosen entity mention
                entity_tokens = [t.lower() for t in chosen_entity.split()]

                # Replace the placeholder with the tokenized entity
                filled_text.extend(entity_tokens)

                # Create IOB tags for each token in the entity mention
                new_iob.append(f'B-{placeholder}')
                new_iob.extend([f'I-{placeholder}'] * (len(entity_tokens) - 1))

            elif placeholder == "YEAR":
                # Generate a random year between 1950 and 2023
                random_year = random.randint(1950, 2023)
                
                # Add the year as a token
                filled_text.append(str(random_year))
                
                # Add the 'O' tag since YEAR is an outside token
                new_iob.append('O')

            else:
                raise ValueError(f"Placeholder '{placeholder}' not found in entity_dict.")
        else:
            # It's a normal token, just append it with an 'O' tag
            filled_text.append(token)
            new_iob.append('O')

    return filled_text, new_iob

def apply_fill_to_templates(df: pd.DataFrame, entities: dict, key: str) -> pd.DataFrame:
    """_summary_
    Args:
        df (pd.DataFrame): dataframe with templates
        entities (dict): entities 
        key (str): key of group
    Returns:
        pd.DataFrame: dataframe with filled entities
    """
    # Helper function to fill a single row's template
    def fill_row_template(row):
        template = row['TEXT_template']  # The template to fill
        n_artists_needed = len([s for s in template if s == "[Artist]"])
        n_woas_needed = len([s for s in template if s == "[WoA]"])

        candidates = entities[key]
        entity = copy.deepcopy(candidates[random.randint(0, len(candidates) - 1)])  # Copy the selected entity
        entity["set_id"] = [entity["set_id"]]
        n_artists = len(entity.get("Artist")) if entity.get("Artist") else 0
        n_woa = len(entity.get("WoA")) if entity.get("WoA") else 0
        while n_artists < n_artists_needed or n_woa < n_woas_needed:
            other_entity = candidates[random.randint(0, len(candidates) - 1)]
            entity["set_id"] += [other_entity["set_id"]]
            entity["Artist"] += other_entity["Artist"]
            entity["WoA"] += other_entity["WoA"]
            n_artists = len(entity["Artist"])
            n_woa = len(entity["WoA"])

        filled_text, IOBs = fill_template_with_entities(template, entity)
        return list(set(entity["set_id"])), filled_text, IOBs  # Return the filled template as a string

    # Apply the function to every row and return a pandas Series with the filled templates
    filled_templates = df.apply(fill_row_template, axis=1)
    return filled_templates

def fill_templates(df: pd.DataFrame, entities: dict) -> pd.DataFrame:
    """Fill into templates
    Args:
        df (pd.DataFrame): dataframe with templates
        entities (dict): dict with entities
    Returns:
        pd.DataFrame: filled templates
    """
    df_concat = pd.DataFrame()

    for key in entities.keys():
    # Call the function
        series = apply_fill_to_templates(df, entities, key)
        df_sub = pd.DataFrame(series.tolist(), columns=['set_id', 'TEXT', 'IOB'])
        df_sub["key"] = key.lower()
        df_concat = pd.concat([df_concat, df_sub])
    return df_concat    
    
def write_stratified_dataset(df: pd.DataFrame, N: int, output_path: str):
    """Write stratified datasets to output path.
    Args:
        df (pd.DataFrame): dataframe with filled templates
        N (int): number of templates
        output_path (str): output path
    """
    m = N // 2

    for model in get_models(df):
        model_output_path = os.path.join(output_path, model)
        os.makedirs(model_output_path, exist_ok=True)
        keys = list(set([key for key in df.key if model in key or key == "post_cutoff"]))

        for i, key in enumerate(keys):
            model_key_output = os.path.join(model_output_path, key.replace("fmt_" + model + "_", ""))
            os.makedirs(model_key_output, exist_ok=True)

            test_set1 = df.loc[df.key == key].iloc[:m]
            set_ids1 = set(test_set1.set_id.explode().to_list())
            test_set2 = df.loc[df.key == key].iloc[m:]
            set_ids2 = set(test_set2.set_id.explode().to_list())
            
            train_set1 = df.iloc[m:]
            train_set1 = train_set1.loc[~train_set1.set_id.isin(set_ids1)]
            train_set2 = df.iloc[:m]
            train_set2 = train_set2.loc[~train_set2.set_id.isin(set_ids2)]

            model_key_output1 = os.path.join(model_key_output, f"dataset1")
            model_key_output2 = os.path.join(model_key_output, f"dataset2")

            os.makedirs(model_key_output1, exist_ok=True)
            os.makedirs(model_key_output2, exist_ok=True)

            if not os.path.isfile(os.path.join(model_key_output1, "test.IOB")):
                write_IOB(train_set1, os.path.join(model_key_output1, "train.IOB"), "IOB")
                write_IOB(test_set1, os.path.join(model_key_output1, "test.IOB"), "IOB")

            if not os.path.isfile(os.path.join(model_key_output2, "test.IOB")):
                write_IOB(train_set2, os.path.join(model_key_output2, "train.IOB"), "IOB")
                write_IOB(test_set2, os.path.join(model_key_output2, "test.IOB"), "IOB")

            model_key_output1p1 = os.path.join(model_key_output + "_perturb1", f"dataset1")
            model_key_output2p1 = os.path.join(model_key_output + "_perturb1", f"dataset2")
            os.makedirs(model_key_output1p1, exist_ok=True)
            os.makedirs(model_key_output2p1, exist_ok=True)
            model_key_output1p2 = os.path.join(model_key_output + "_perturb2", f"dataset1")
            model_key_output2p2 = os.path.join(model_key_output + "_perturb2", f"dataset2")
            os.makedirs(model_key_output1p2, exist_ok=True)
            os.makedirs(model_key_output2p2, exist_ok=True)

            if not os.path.isfile(os.path.join(model_key_output1p1, "test.IOB")):
                write_IOB(make_perturb_dataset(train_set1), os.path.join(model_key_output1p1, "train.IOB"), "IOB")
                write_IOB(make_perturb_dataset(test_set1), os.path.join(model_key_output1p1, "test.IOB"), "IOB")

            if not os.path.isfile(os.path.join(model_key_output2p1, "test.IOB")):
                write_IOB(make_perturb_dataset(train_set2), os.path.join(model_key_output2p1, "train.IOB"), "IOB")
                write_IOB(make_perturb_dataset(test_set2), os.path.join(model_key_output2p1, "test.IOB"), "IOB")

            if not os.path.isfile(os.path.join(model_key_output1p2, "test.IOB")):
                write_IOB(make_perturb_dataset(train_set1, combine=True), os.path.join(model_key_output1p2, "train.IOB"), "IOB")
                write_IOB(make_perturb_dataset(test_set1, combine=True), os.path.join(model_key_output1p2, "test.IOB"), "IOB")

            if not os.path.isfile(os.path.join(model_key_output2p2, "test.IOB")):
                write_IOB(make_perturb_dataset(train_set2, combine=True), os.path.join(model_key_output2p2, "train.IOB"), "IOB")
                write_IOB(make_perturb_dataset(test_set2, combine=True), os.path.join(model_key_output2p2, "test.IOB"), "IOB")

def load_memorization_data(input_path: str, shs_path: str) -> pd.DataFrame:
    """Get enriched memorization dataframe
    Args:
        input_path (str): path to memorization data
        shs_path (str): path to SecondHandsongs metadata
    Returns:
        pd.DataFrame: enriched memorization dataframe 
    """
    
    df = pd.read_json(input_path, lines=True, orient="records")

    def transform_to_multiindex(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        def strip(s):
            return s.rstrip('_- ').lstrip('_- ')
        def split_col(col):
            if col.endswith(suffix):
                return (strip(col[:-len(suffix)]), strip(suffix))
            else:
                return (strip(col), '')

        new_columns = [split_col(col) for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(new_columns)
        return df

    df = transform_to_multiindex(df, '_Correctness')

    df_shs = pd.read_json(shs_path, lines=True, orient="records")
    df_shs = df_shs[
        ["set_id", "work_id", "perf_id", "perf_artist", "perf_title"]].rename(
        columns={"perf_artist": "Artist", "perf_title": "WoA"}
    )

    current_columns = df_shs.columns
    new_columns = pd.MultiIndex.from_product([current_columns, ['']])
    df_shs.columns = new_columns
    df_shs = df_shs.dropna(subset=[('Artist', ''), ( 'WoA', '')])
    df_shs.WoA = df_shs.WoA.apply(lambda x: [random.sample(get_title_variations(x), 1)[0].lower()])
    df_shs.Artist = df_shs.Artist.apply(lambda x: get_performer_variations(x))

    # TODO: check join on set_id only vs. set_id, work_id, perf_id
    df = pd.merge(df, df_shs, on="set_id", how="left")
    return df

def get_models(df: pd.DataFrame) -> List[str]:
    """Get models from memorization dataframe
    Args:
        df (pd.DataFrame): memorization dataframe
    Returns:
        List[str]: list of models
    """
    return df.columns.get_level_values(level=0).unique()

def get_memorization_entities(df: pd.DataFrame, entitites: dict) -> dict:
    """Fill memorization entities with data from memorization test.
    Args:
        df (pd.DataFrame): memorization dataframe
        entitites (dict): entity dict
    Returns:
        dict: entities with data from memorization test 
    """
    for model in get_models(df):
        seen, unseen = get_seen_unseen_entities(df, model)
        entitites[f"fmt_{model}_seen"] = seen
        entitites[f"fmt_{model}_unseen"] = unseen
    return entitites

def get_seen_unseen_entities(df: pd.DataFrame, model: str) -> Tuple[List[dict], List[dict]]:
    """Get seen and unseen entities for model.
    Args:
        df (pd.DataFrame): memorization dataframe
        model (str): model string
    Returns:
        Tuple[List[dict], List[dict]]: seen and unseen entities
    """
    # TODO fix masks
    mask_unseen = df.loc[:,[model]].T.sum() == 0
    mask_seen = df.loc[:,[(model, "AW1: Correct"),(model, "AW1: Correct")]].T.sum() == 2
    
    df_seen = df_seen.loc[mask_seen, ["set_id", "Artist", "WoA"]]
    df_seen.columns = ["set_id", "Artist", "WoA"]
    seen = df_seen.to_dict(orient="records")

    df_unseen = df.loc[mask_unseen, ["set_id", "Artist", "WoA"]]
    df_unseen.columns = ["set_id", "Artist", "WoA"]
    unseen = df_unseen.to_dict(orient="records")
    return seen, unseen
    
def get_postcutoff_entities(input_file: str) -> List[str]:
    """Get entities of post-cutoff data from MusicBrainz.
    Args:
        input_file (str): File to processed data from musicbrainz
    Returns:
        List[str]: list of entities to fill tempaltes
    """
    df = pd.read_json(input_file, lines=True, orient="records")
    ents = []
    for row in df[["name", "release_title2"]].dropna().to_dict(orient="records"):
        e = {}
        e["set_id"] = -1
        e["Artist"] = get_performer_variations(row["name"].lower())
        e["WoA"] = get_title_variations(row["release_title2"].lower())
        ents.append(row)
    return ents
    
def make_template_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Make columns for clozes.
    Args:
        df (pd.DataFrame): dataset
    Returns:
        pd.DataFrame: _description_
    """
    def process_sequence(iob_tags, text_tokens):
        processed_iob = []
        processed_text = []
        current_class = None

        for iob, token in zip(iob_tags, text_tokens):
            if iob.startswith('B-'):
                current_class = iob[2:]  # Get the class name after 'B-'
                processed_iob.append(current_class)
                processed_text.append(f'[{current_class}]')
            elif iob.startswith('I-') and current_class:
                # Continue the current class, but skip it in output
                continue
            else:
                # It's an 'O' tag or something else
                current_class = None
                processed_iob.append('O')
                processed_text.append(token)
        
        return processed_iob, processed_text

    # Apply the process_sequence function row-wise
    df[['IOB_template', 'TEXT_template']] = df.apply(
        lambda row: process_sequence(row["IOB"], row["TEXT"]), axis=1, result_type='expand'
    )
    
    def replace_years(lst):
        return [re.sub(r'19\d{2}|20\d{2}', '[YEAR]', s) if isinstance(s, str) else s for s in lst]

    df.TEXT_template = df.TEXT_template.apply(replace_years)
    return df.drop_duplicates(subset=["TEXT_template"])
    
def load_dataset(input_dir: str) -> pd.DataFrame:
    """Load dataset as a pandas DataFrame.
    Args:
        input_dir (str): path to dataset as k-fold
    Returns:
        pd.DataFrame: dataset dataframe
    """
    df = pd.read_json(os.path.join(input_dir, "data.jsonl"), lines=True)
    df["TEXT"] = None
    df["IOB"] = None
    
    for subset_id in df.subset.unique():
        subset_dir = os.path.join(input_dir, f"dataset{subset_id}")
        iob_file = os.path.join(subset_dir, "test.IOB")
        texts, iobs = read_IOB_file(iob_file)

        for i, text in enumerate(texts):
            df.loc[(df.subset == subset_id) & (df["index"] == i), 'TEXT'] = ' '.join(text)
            df.loc[(df.subset == subset_id) & (df["index"] == i), 'IOB'] = ' '.join(iobs[i])   
    
    assert df.TEXT.isna().sum() == 0 and df.IOB.isna().sum() == 0, "Mismatch between metadata and subset files"    
    
    df["has_WoA"] = df.IOB.apply(lambda x: "B-WoA" in x)
    df["has_Artist"] = df.IOB.apply(lambda x: "B-Artist" in x)
    return df

def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Only retain where WoA and Artist class label and outside context.
    Args:
        df (pd.DataFrame): dataframe
    Returns:
        pd.DataFrame: template dataframe
    """
    df_template = df[df.has_WoA | df.has_Artist]
    df_template = df_template[df_template.IOB.apply(lambda x: "O" in x)]
    return df_template


def perturb_characters(s: str, n: int) -> str:
    """
    Perturb a specific number of characters in the input text.
    
    Args:
        s (str): The original text to perturb.
        n (int): The number of characters to perturb.
        
    Returns:
        str: The perturbed text.
    """
    chars = list(s)
    text_length = len(chars)

    if num_chars_to_perturb > text_length:
        num_chars_to_perturb = text_length

    for _ in range(n):
        ptype = random.choice(["substitution", "deletion", "insertion"])
        i = random.randint(0, len(chars) - 1)
        
        if ptype == "substitution":
            # Replace the character with a random one
            chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
        elif ptype == "deletion" and len(chars) > 1:
            # Remove the character
            chars.pop(i)
        elif ptype == "insertion":
            # Insert a random character
            chars.insert(i, random.choice("abcdefghijklmnopqrstuvwxyz"))
    
    return ''.join(chars)


def perturb_abbrv(s: str) -> str:
    """Abbreviation perturbation
    Args:
        s (str): input string
    Returns:
        str: perturbed string
    """
    words = s.split()
    
    if len(words) > 1:
        abbreviation = ''.join([word[0].lower() for word in words])
        return abbreviation
    else:
        return s    

def perturb_tokens(s, num_perturbations=1):
    """
    Perturb tokens in the input text.
    
    Args:
        text (str): The original text to perturb.
        perturbation_strength (float): The proportion of tokens to perturb (0 to 1).
        
    Returns:
        str: The perturbed text.
    """
    tokens = s.split()
    
    for _ in range(num_perturbations):
        ptype = random.choice(["deletion", "shuffle"])
        i = random.randint(0, len(tokens) - 1)
        
        if ptype == "deletion" and len(tokens) > 1:
            tokens.pop(i)
        elif ptype == "shuffle" and len(tokens) > 1:
            j = random.randint(0, len(tokens) - 1)
            while j == i:
                j = random.randint(0, len(tokens) - 1)

            tokens[i], tokens[j] = tokens[j], tokens[i]
    
    return ' '.join(tokens)


def apply_class_perturbation(s: str, cls: str, p: float, combine: bool) -> str:
    """Perturb the input string based on the entity class.
    Args:
        s (str): input string
        cls (str): entity class
        p (float): perturbation probability
        combine (bool): whether to combine perturbation
    Returns:
        str: perturbed string
    """
    if combine:
        if cls == "WoA":
            if random.random() < p:
                s = perturb_tokens(s, 1)  # Apply token-level perturbation
            if random.random() < p:
                s = perturb_characters(s, 1)  # Apply character-level perturbation
        elif cls == "Artist":
            if random.random() < p:
                if random.choice([True, False]):
                    s = perturb_abbrv(s)  # Abbreviation perturbation
                else:
                    s = perturb_tokens(s, 1)  # Token-level perturbation
            if random.random() < p:
                s = perturb_characters(s, 1)  # Apply character-level perturbation
    else:
        if random.random() < p:
            if random.choice([True, False]):
                s = perturb_tokens(s, 1)  # Token-level perturbation
            else:
                s = perturb_characters(s, 1)  # Character-level perturbation
    return s

def make_perturb_dataset(df: pd.DataFrame, p: float = 0.5, combine: bool = False) -> pd.DataFrame:
    """
    Apply perturbations to the dataframe containing text and IOB tags.
    
    Args:
        df (pd.DataFrame): The input dataframe containing 'TEXT' and 'IOB' columns.
        p (float): Probability of applying perturbation per utterance.
        combine (bool): Whether to combine perturbation types or apply a single perturbation.
    Returns:
        pd.DataFrame: A dataframe with perturbated 'TEXT' and 'IOB' columns.
    """
    # Create a copy of the dataframe to avoid modifying the original
    perturbed_df = df.copy()
    
    perturb_words = []
    perturb_iobs = []

    # Iterate over each row in the dataframe
    for i, row in perturbed_df.iterrows():
        words = row["TEXT"]
        iob_tags = row["IOB"]
        
        # Initialize empty lists to store the new perturbed words and IOB tags
        new_words = []
        new_iob_tags = []

        current_entity = None
        current_words = []
        current_tags = []
        
        for word, tag in zip(words, iob_tags):
            if tag == 'O':
                # Apply perturbation to the previous entity, if necessary
                if current_entity:
                    perturbed_entity = apply_class_perturbation(' '.join(current_words), current_entity, p, combine)
                    new_words.extend(perturbed_entity.split())
                    # Generate new IOB tags for the perturbed entity
                    new_tags = [f'I-{current_entity}'] * len(perturbed_entity.split())
                    new_tags[0] = f'B-{current_entity}'
                    new_iob_tags.extend(new_tags[:len(current_tags)])  # Adjust to match the tags

                # Add the current non-entity word and tag
                new_words.append(word)
                new_iob_tags.append(tag)
                current_entity = None
                current_words = []
                current_tags = []
            elif tag.startswith('B-'):
                # Encounter a new entity
                if current_entity:
                    perturbed_entity = apply_class_perturbation(' '.join(current_words), current_entity, p, combine)
                    new_words.extend(perturbed_entity.split())
                    new_tags = [f'I-{current_entity}'] * len(perturbed_entity.split())
                    new_tags[0] = f'B-{current_entity}'
                    new_iob_tags.extend(new_tags[:len(current_tags)])
                
                # Start a new entity
                current_entity = tag[2:]  # Extract the class (WoA, Artist)
                current_words = [word]
                current_tags = [tag]
            elif tag.startswith('I-') and current_entity == tag[2:]:
                # Continue with the current entity
                current_words.append(word)
                current_tags.append(tag)

        # Finalize any remaining entity at the end of the row
        if current_entity:
            perturbed_entity = apply_class_perturbation(' '.join(current_words), current_entity, p, combine)
            new_words.extend(perturbed_entity.split())
            new_tags = [f'I-{current_entity}'] * len(perturbed_entity.split())
            new_tags[0] = f'B-{current_entity}'
            new_iob_tags.extend(new_tags[:len(current_tags)])

        # Replace the row's words and tags with the perturbed versions
        perturb_words.append(new_words)
        perturb_iobs.append(new_iob_tags)
    
    perturbed_df["TEXT"] = perturb_words
    perturbed_df["IOB"] = perturb_iobs
    
    return perturbed_df

def parse_args():
    parser = argparse.ArgumentParser(description='Generate cloze dataset from input directory.')
    parser.add_argument('--input_dir', type=str, default="data/dataset/reddit+shsyt", help='Path to the directory with the dataset in five fold cross validation.')
    parser.add_argument('--debuts_file', type=str, default="data/intermediate/debut_performers.jsonl",  help='Path to file with debut performers as crawled from MusicBrainz.')
    parser.add_argument('--shs_file', type=str, default="data/source/shs100k2_rich.jsonl",  help='Path to file with debut performers as crawled from MusicBrainz.')
    parser.add_argument('--memorization_file', type=str, default="data/intermediate/memorization.jsonl",  help='Path to file with debut performers as crawled from MusicBrainz.')
    parser.add_argument('--output_dir', type=str, default="data/dataset/reddit+shsyt_cloze2",  help='Path to the directory where to put the output cloze dataset.')

    return parser.parse_args()

def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # make template columns
    df_template = filter_df(load_dataset(input_dir))
    df_template = make_template_cols(df_template)
    
    # make dict with entities to fill into templates
    entities = {}
    print("Collect Debut Performers...")
    entities["post_cutoff"] = get_postcutoff_entities(args.debuts_file)
    print("Done.")


    print("Collect Memorization Test Results...")
    df_memorization = load_memorization_data(args.memorization_file, 
                                             args.shs_file)
    entities = get_memorization_entities(df_memorization, entities)
    print("Done.")
    
    # fill
    print("Fill into templates...")
    df_filled = fill_templates(df_template, entities)
    write_stratified_dataset(df=df_filled, N=len(df_template), output_path=output_dir)
    print("Done.")

if __name__ == '__main__':
    main()