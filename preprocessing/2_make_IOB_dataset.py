import argparse
import pandas as pd
from itertools import combinations
from typing import Tuple, Dict, List, Callable
from tqdm import tqdm
from Utils import (SONG_ATTRS, CLASS_ATTRS, 
                   B_PREFIX, I_PREFIX, O_LABEL, 
                   BASELINE_NAMES, simplify_string, strip_list_special_chars, 
                   char_idx_to_word_idx, find_sublist_indices)
from rapidfuzz.fuzz import partial_ratio_alignment
import numpy as np


def find_ent_utterance(text: str, start_idx: int, end_idx: int) -> List[str]:

    def find_closest_nonspace_idx(s: str, char_idx: int, direction: str) -> int:
        """
        Args:
            s (str): the string
            char_idx: (int): initial index
            direction (str): left or right?
        Returns:
            int: the closest index with not space as char, or -1 if not found
        """
        if s[char_idx] != " ":
            return char_idx
        assert direction in ["left", "right"], "Direction must be left or right!"
        
        if direction == "left":
            # Iterate from char_idx to the left end of the string
            for i in range(char_idx - 1, -1, -1):
                if s[i] != ' ':
                    return i
        elif direction == "right":
            # Iterate from char_idx to the right end of the string
            for i in range(char_idx + 1, len(s)):
                if s[i] != ' ':
                    return i      
        return -1  
    
    idx_first, idx_last = find_closest_nonspace_idx(text, start_idx, "right"), find_closest_nonspace_idx(text, end_idx - 1, "left")

    def handle_cutoff_words(ent: List[str], first_actual: str, last_actual: str, min_frac: float = 0.5) -> List[str]:
        """Handle cutoff words.
        Args:
            text (List[str]): 
            first_actual (str): 
            last_actual (str): 
            min_frac (float): minimum fraction of word
        Returns:
            List[str]: 
        """
        first_extracted, last_extracted = ent[0], ent[-1]

        if first_extracted != first_actual:
            if len(first_extracted) < min_frac * len(first_actual):
                ent = ent[1:]
        
        if last_extracted != last_actual:
            if len(last_extracted) < min_frac * len(last_actual):
                ent = ent[:-1]
        return ent

    tokens = text.split()
    ent = text[idx_first:idx_last + 1].split()

    first_word, last_word = tokens[char_idx_to_word_idx(text, idx_first)], tokens[char_idx_to_word_idx(text, idx_last)]
    ent = handle_cutoff_words(ent, first_word, last_word)
    return ent

def find_word_partial(text1: str, text2: str, start: int = 0, min_r: int = 90) -> Tuple[Tuple[int, int], float]:
    """Find text2 (shorter string) in text1 (eg. YT metadata) with partial alignment.
    Args:
        text1 (str): longer string
        text2 (str): shorter string
        min_r (int): minimum ratio required
        start (int, optional): start index. Defaults to 0.
    Returns:
        Tuple[Tuple[int, int], float]: start and end index and score
    """

    _text1 = ' '.join(text1.split()[start:])
    if not (len(_text1) < len(text2) or start == -1 or text2 == '' or _text1 == ''):
        # find partial alignment with rapidfuzz
        al = partial_ratio_alignment(text2, _text1)
        if al.score >= min_r:
            ent = find_ent_utterance(_text1, al.dest_start, al.dest_end)
            # find start index
            if len(ent) > 0:
                # strip special chars
                ent = strip_list_special_chars(ent)

                start_inds = find_sublist_indices(_text1.split(), ent)
                if len(start_inds) > 0:
                    return ((start_inds[0] + start, start_inds[0] + len(ent) + start), al.score)
    return ((-1, -1), None) 

def overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """Compute overlap between two spans.
    Args:
        span1 (Tuple[int, int]): 
        span2 (Tuple[int, int]): 
    Returns:
        bool: if overlapping
    """
    (s1_start, s1_end) = span1
    (s2_start, s2_end) = span2
    return (s1_end > s2_start and s2_start >= s1_start) or (s2_end > s1_start and s1_start >= s2_start) or (s1_start == s2_start)

def span_len(span) -> int:
    return abs(span[0] - span[1])

# resolve overlapping 
def __resolve_span_overlaps(ent_spans: Dict[Tuple[int, int], str]) -> Dict[Tuple[int, int], str]:
    """Resolve span overlaps by retaining the bigger span.
    Args:
        ent_spans (Dict[Tuple[int, int], str]): Keys: spans, Values: entity tags 
    Returns:
        dict: resolved ent_spans
    """
    for span_pair in combinations(ent_spans.items(), 2):
        ((span1, (_, _)), (span2, (_, _))) = span_pair
        if overlap(span1, span2) and ent_spans.get(span1) and ent_spans.get(span2):
            if span_len(span1) >= span_len(span2):
                del ent_spans[span2]
            else:
                del ent_spans[span1]
    return ent_spans

def __spans_to_taglist(text: str, ent_spans: Dict[Tuple[int, int], str]) -> Tuple[List[str], List[float]]:
    """Generate a list of N where N is the number of words in text with the span labels obtained before.
    Args:
        ent_spans (Dict[Tuple[int, int], str]): Keys: spans, Values: entity tags. 
    Returns:
        Tuple[List[str], List[float]]: list with IOB tags and scores
    """
    tag_list = [O_LABEL for i in range(len(text.split()))]
    score_list = [None for i in range(len(text.split()))]

    for span, (ent_tag, score) in ent_spans.items():
        
        start_idx = span[0]

        # first token
        tag_list[start_idx] = B_PREFIX + ent_tag
        score_list[start_idx] = score

        # remaining tokens
        for idx in range(start_idx + 1, span[1]):
            tag_list[idx] = I_PREFIX + ent_tag

    return (tag_list, score_list)


def make_taglist(item: pd.Series, ent_names: List[str], baseline_name: bool, all: bool, 
                 min_r: int) -> List[str]:
    """Creates a tag list with IOB tags for NER based on yt metadata (yt_processed) in the dataframe item.
    Args:
        item (pd.Series): Row in the dataframe.
        ent_names (List[str]): list of entity names
        baseline_name (bool): whether to change entity names to coarse attributes from the baseline approach
        all (bool): Whether to search for all occurances or only the first.
        min_r (int): minimum ratio for partial ratio. If none, exact matching.
    Returns:
        List[str]: list with IOB tags
    """
    text = item["yt_processed"]
    # simplify for more robust matching
    match_text = simplify_string(text)

    ent_spans = {}
    for ent_name in ent_names:
        # assume list (split performers), otherwise, make one element list
        ents = item[ent_name]
        if type(ents) == str:
            ents = [ents]

        # for each entity (eg. performer)
        for ent in ents:

            start = 0

            # simplify for more robust matching
            match_ent = simplify_string(ent)

            # all occurances
            while start >= 0:
                
                span, score = find_word_partial(match_text, match_ent, start, min_r)

                # stop if entity is not found at all
                if span[0] == -1:
                    break
                
                # add to mapping only if entity is new
                if not ent_spans.get(span): 
                    
                    # change entity class name
                    ent_tag = ent_name.replace("_processed", "")
                    if baseline_name:
                        ent_tag = BASELINE_NAMES[ent_tag]
                    
                    ent_spans[span] = (ent_tag, score)

                start = span[1] + 1 if all else -1

    ent_spans = __resolve_span_overlaps(ent_spans)

    return __spans_to_taglist(match_text, ent_spans)


def __get_max_indel(row: pd.Series, ent_name: str = "WoA") -> float:
    """Get maximum indel distance for matched entity.
    Args:
        row (pd.Series): row in dataframe
        ent_name (str, optional): entity name. Defaults to "WoA".
    Returns:
        float: _description_
    """
    def __get_ent_inds(l: List[str], ent_name: str = "WoA") -> List[int]:
        """Given a list, get indices with ent_name tags
        Args:
            l (List[str]): 
            ent_name (str, optional): entity name. Defaults to "WoA".
        Returns:
            List[int]: list of indices
        """
        return [i for (i, e) in enumerate(l) if ent_name in e]
    dists = [row.IOB_Indel[i] for i in __get_ent_inds(row.IOB, ent_name) if row.IOB_Indel[i] is not None]
    if len(dists) > 0:
        return max(dists)
    return None

def attach_segment(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Attach categories of difficulty to dataframe.
    Args:
        data (pd.DataFrame): 
        col_name (str): 
    Returns:
        pd.DataFrame: 
    """
    series_woa = data.apply(lambda x: __get_max_indel(x, "WoA"), axis=1)
    series_artist = data.apply(lambda x: __get_max_indel(x, "Artist"), axis=1)

    data[col_name] = None
    # both_100
    mask_both_100 = series_woa.apply(lambda x: x == 100) & series_artist.apply(lambda x: x == 100)
    data.loc[mask_both_100, col_name] = "both_100"
    # both nan
    mask_both_nan = series_artist.isna() & series_woa.isna()
    data.loc[mask_both_nan, col_name] = "both_nan"
    # woa nan
    mask_woa_nan = ~mask_both_nan & series_woa.isna()
    data.loc[mask_woa_nan, col_name] = "WoA_nan"
    # artist nan
    mask_artist_nan = ~mask_both_nan & series_artist.isna()
    data.loc[mask_artist_nan, col_name] = "Artist_nan"
    # other
    mask_medium = ~mask_both_100 & ~mask_both_nan & ~mask_woa_nan & ~mask_artist_nan
    data.loc[mask_medium, col_name] = "medium"
    return data

def retag_matches(text_list: np.ndarray, tag_list: np.ndarray) -> np.ndarray:
    """
    Retag the tag_list array with the correct IOB tags based on all matching entities.

    Args:
        text_list (np.ndarray): The array of text_list.
        tag_list (np.ndarray): The array of IOB tags.

    Returns:
        np.ndarray: The updated array of IOB tags.
    """

    entities = []
    i = 0
    while i < len(text_list):
        if tag_list[i].startswith('B-'):
            entity_words = []
            entity_tags = []
            while i < len(text_list) and (tag_list[i].startswith('I-') or (tag_list[i].startswith('B-') and not entity_words)):
                entity_words.append(text_list[i])
                entity_tags.append(tag_list[i])
                i += 1
            entities.append((np.array(entity_words), np.array(entity_tags)))
        else:
            i += 1
    
    for entity_words, entity_tags in entities:
        indices = find_sublist_indices(text_list, entity_words)
        for index in indices:
            tag_list[index:index + len(entity_words)] = entity_tags
    
    return tag_list

def main():

    args = parse_args()

    data = pd.read_parquet(args.input)

    ent_names = [ent + '_processed' for ent in SONG_ATTRS if ent + '_processed' in data.columns]

    exact = args.min_r == None
    if exact:
        min_r = 100
    else:
        min_r = args.min_r
    print(f"Creating dataset: ALL={args.all}; min_r={min_r}")

    tqdm.pandas()
    data["TEXT"] = data.yt_processed.progress_apply(lambda x: simplify_string(x).split())

    # 1. generate IOB tags
    series = data.progress_apply(make_taglist, args=(ent_names, args.baseline_names, args.all, min_r), axis=1)
    data["IOB"] = series.apply(lambda x: x[0])
    data["IOB_Indel"] = series.apply(lambda x: x[1])

    # 2. attach partitions
    data = attach_segment(data, "part")

    # 3. retag for consistency
    # manually retag match-based for partial
    print("Retag IOBs...")
    data["IOB"] = data.progress_apply(lambda x: retag_matches(x.TEXT, x.IOB), axis=1)

    data.to_parquet(args.output)

def parse_args():
    parser = argparse.ArgumentParser(description='Make IOB tag list for NER task.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output parquet file.')
    parser.add_argument('-m', '--min_r', type=int, default=80, help='Minimum partial ratio to use for matching. If None or 100, exact matching is performed.')
    parser.add_argument('--baseline_names', action='store_true', help='Whether to change entity class name to the ones used in the baseline approach.')
    parser.add_argument('--all', action='store_true', help='Whether to find all or only the first occurance per entity in the string.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()