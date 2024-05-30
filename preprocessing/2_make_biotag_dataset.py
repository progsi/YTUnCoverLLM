import argparse
import pandas as pd
from itertools import combinations
from typing import Tuple, Dict, List, Callable
from tqdm import tqdm
from Utils import (SONG_ATTRS, CLASS_ATTRS, 
                   B_PREFIX, I_PREFIX, O_LABEL, 
                   BASELINE_NAMES, simplify_string, isolate_special_chars)
from rapidfuzz.fuzz import partial_ratio_alignment


def find_word(text1: str, text2: str, start: int = 0) -> int:
    """Like string.find but for word-level search.
    Args:
        text1 (str): The long text.
        text2 (str): The short text.
        start (int): where to start
    Returns:
        int: start index
    """
    list1 = text1.split()[start:]
    list2 = text2.split()

    # If the second list is longer than the first, return -1
    if len(list1) < len(list2):
        return -1
    # Iterate through the longer list to find the position of the shorter list
    for i in range(len(list1) - len(list2) + 1):
        if list1[i:i+len(list2)] == list2:
            return i + start
    return -1

def find_word_start_end(text1: str, text2: str, start: int = 0) -> Tuple[int, int]:
    """Like find_word but with start and end index.
    Args:
        text1 (str): The long text.
        text2 (str): The short text.
        start (int): where to start
    Returns:
        Tuple[int, int]: start and end index
    """
    start_idx = find_word(text1, text2, start)
    if start_idx == -1 or text2 == '' or text1 == '':
        return (-1, -1)
    end_idx = start_idx + len(text2.split()) - 1
    return (start_idx, end_idx)

def __char_index_to_word_index(s: str, idx: int) -> int:
    """Helper to transform char index in string to word index (after split by space).
    Args:
        s (str): word index
        idx (int): char index
    Returns:
        int: word level index
    """
    cur_idx = 0
    for w_idx, word in enumerate(s.split()):
        # consider length and space
        cur_len = len(word) + 1
        if cur_idx <= idx < cur_idx + cur_len:
            return w_idx
        cur_idx += cur_len
    return -1

def __char_span_to_word_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """
    For a given span of char indices (start, end) in a string, find the corresponding word index span
    based on the string indices.
    Args:
        text (str): string
        start (int): span start index in non-split string
        end (int): span end index in non-split string
    Returns:
        Tuple[int, int]: span start and end indices
    """
    return (__char_index_to_word_index(text, start), __char_index_to_word_index(text, end))


def find_word_partial(text1: str, text2: str, start: int = 0, min_r: int = 90) -> Tuple[int, int]:
    """Find text2 (shorter string) in text1 (eg. YT metadata) with partial alignment.
    Args:
        text1 (str): longer string
        text2 (str): shorter string
        min_r (int): minimum ratio required
        start (int, optional): start index. Defaults to 0.
    Returns:
        Tuple[int, int]: start and end index
    """

    _text1 = ' '.join(text1.split()[start:])
    if not (len(_text1) < len(text2) or start == -1 or text2 == '' or _text1 == ''):
        # find partial alignment with rapidfuzz
        al = partial_ratio_alignment(text2, _text1)
        if al.score >= min_r:
            char_start = al.dest_start
            char_end = al.dest_end

            (word_start, word_end) = __char_span_to_word_span(_text1, char_start, char_end)

            return (word_start + start, word_end + start)
    return (-1, -1)

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
    return (s1_end >= s2_start and s2_start >= s1_start) or (s2_end >= s1_start and s1_start >= s2_start)

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
        ((span1, ent_name1), (span2, ent_name2)) = span_pair
        if overlap(span1, span2) and ent_spans.get(span1) and ent_spans.get(span2):
            if span_len(span1) >= span_len(span2):
                del ent_spans[span2]
            else:
                del ent_spans[span1]
    return ent_spans

def __spans_to_taglist(text: str, ent_spans: Dict[Tuple[int, int], str]) -> List[str]:
    """Generate a list of N where N is the number of words in text with the span labels obtained before.
    Args:
        ent_spans (Dict[Tuple[int, int], str]): Keys: spans, Values: entity tags. 
    Returns:
        List[str]: list with BIO tags
    """
    tag_list = [O_LABEL for i in range(len(text.split()))]

    for span, ent_tag in ent_spans.items():
        
        start_idx = span[0]

        # first token
        tag_list[start_idx] = B_PREFIX + ent_tag

        # remaining tokens
        for idx in range(start_idx + 1, span[1] + 1):
            tag_list[idx] = I_PREFIX + ent_tag

    return tag_list

def __tag_missing(text_list: List[str], tag_list: List[str]) -> List[str]:
    """Since it is not guranteed that partial ratio matches all utterances, 
    here we tag them afterwards based on matching words.
    Args:
        text_list (List[str]): words in the sequence
        tag_list (List[str]): initial tag list
    Returns:
        List[str]: tag list
    """
    for i, tag in enumerate(tag_list):
        if tag != O_LABEL:
            for j, word in enumerate(text_list):
                if (i != j) and (word == text_list[i]) and (tag_list[j] == O_LABEL):
                    tag_list[j] = tag
    return tag_list

def make_taglist(item: pd.Series, ent_names: List[str], baseline_name: bool, all: bool, 
                 min_r: int = None) -> List[str]:
    """Creates a tag list with BIO tags for NER based on yt metadata (yt_processed) in the dataframe item.
    Args:
        item (pd.Series): Row in the dataframe.
        ent_names (List[str]): list of entity names
        baseline_name (bool): whether to change entity names to coarse attributes from the baseline approach
        all (bool): Whether to search for all occurances or only the first.
        min_r (int): minimum ratio for partial ratio. If none, exact matching.
    Returns:
        List[str]: list with BIO tags
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
                
                if min_r == 100 or min_r is None:
                    span = find_word_start_end(match_text, match_ent, start)
                else:
                    span = find_word_partial(match_text, match_ent, start, min_r)

                # stop if entity is not found at all
                if span[0] == -1:
                    break
                
                # add to mapping only if entity is new
                if not ent_spans.get(span): 
                    
                    # change entity class name
                    ent_tag = ent_name.replace("_processed", "")
                    if baseline_name:
                        ent_tag = BASELINE_NAMES[ent_tag]
                    
                    ent_spans[span] = ent_tag

                start = span[1] + 1 if all else -1

    ent_spans = __resolve_span_overlaps(ent_spans)

    return __spans_to_taglist(match_text, ent_spans)

def main():

    args = parse_args()

    data = pd.read_parquet(args.input)

    ent_names = [ent + '_processed' for ent in SONG_ATTRS if ent + '_processed' in data.columns]

    partial = args.min_r in [100, None]

    print(f"Creating dataset: Partial={partial}; ALL={args.all}")

    tqdm.pandas()
    data["TEXT"] = data.yt_processed.progress_apply(lambda x: simplify_string(x).split())

    data["NER_TAGS_PARTIAL"] = data.progress_apply(make_taglist, args=(ent_names, args.baseline_names, args.all, args.min_r), axis=1)
    data["NER_TAGS_EXACT"] = data.progress_apply(make_taglist, args=(ent_names, args.baseline_names, args.all, None), axis=1)
    # TODO: curate the TAGLISTs from partial, so that matching entities have same labels! eg Yesterday found at (7,7) but not at (3,3) 
    data.to_parquet(args.output)

def parse_args():
    parser = argparse.ArgumentParser(description='Make BIO tag list for NER task.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output parquet file.')
    parser.add_argument('-m', '--min_r', type=int, default=80, help='Minimum partial ratio to use for matching. If None or 100, exact matching is performed.')
    parser.add_argument('--baseline_names', action='store_true', help='Whether to change entity class name to the ones used in the baseline approach.')
    parser.add_argument('--all', action='store_true', help='Whether to find all or only the first occurance per entity in the string.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()