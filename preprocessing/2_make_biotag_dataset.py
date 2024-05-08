import argparse
import pandas as pd
from itertools import combinations
from typing import Tuple, Dict, List
from tqdm import tqdm


# order important! Due to overlaps between title and artist strings.
SONG_ATTRS = ["title", "title_perf", "title_work", "performer", "performer_perf", "performer_work"]
CLASS_ATTRS = ["set_id", "ver_id", "yt_id"]

# mapping to classes from the baseline paper (coarse)
BASELINE_NAMES = {
    "title": "WoA", 
    "performer": "Artist", 
    "title_perf": "WoA", 
    "performer_perf": "Artist", 
    "title_work": "WoA", 
    "performer_work": "Artist",
    "ambiguous": "Artist_or_WoA"
}

# for label names
B_PREFIX = "B-"
I_PREFIX = "I-"
O_LABEL = "0"


def find_word(text1: str, text2: str, start: int = 0):
    """Like string.find but for word-level search.
    Args:
        text1 (str): The long text.
        text2 (str): The short text.
        start (int): where to start
    Returns:
        int: start index
    """
    text1 = text1.split()[start:]
    text2 = text2.split()

    # If one list is longer than the other, return -1
    if len(text1) < len(text2):
        return -1
    # Iterate through the longer list to find the position of the shorter list
    for i in range(len(text1) - len(text2) + 1):
        if text1[i:i+len(text2)] == text2:
            return i + start
    return -1

def find_word_start_end(text1: str, text2: str, start: int = 0):
    """Like find_word but with start and end index.
    Args:
        text1 (str): The long text.
        text2 (str): The short text.
        start (int): where to start
    Returns:
        int: start index
    """
    start_idx = find_word(text1, text2, start)
    if start_idx == -1 or text2 == '' or text1 == '':
        return (-1, -1)
    end_idx = start_idx + len(text2.split()) - 1
    return (start_idx, end_idx)

def overlap(span1: Tuple[int, int], span2: Tuple[int, int]):
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

def span_len(span):
    return abs(span[0] - span[1])

# resolve overlapping 
def __resolve_span_overlaps(ent_spans: Dict[Tuple[int, int], str]):
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

def __spans_to_taglist(text: str, ent_spans: Dict[Tuple[int, int], str]):
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

def make_taglist(item: pd.Series, ent_names: List[str], baseline_name: bool, all: bool):
    """Creates a tag list with BIO tags for NER based on yt metadata (yt_processed) in the dataframe item.
    Args:
        item (pd.Series): Row in the dataframe.
        ent_names (List[str]): list of entity names
        baseline_name (bool): whether to change entity names to coarse attributes from the baseline approach
        all (bool): Whether to search for all occurances or only the first.
    Returns:
        List[str]: list with BIO tags
    """
    text = item["yt_processed"].replace("\n", " ")

    ent_spans = {}
    for ent_name in ent_names:
        ent = item[ent_name].replace("\n", " ")
        start = 0
        
        # all occurances
        while start >= 0:

            span = find_word_start_end(text, ent, start)

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

    return __spans_to_taglist(text, ent_spans)

def main():

    args = parse_args()

    data = pd.read_parquet(args.input)

    ent_names = [ent + '_processed' for ent in SONG_ATTRS if ent + '_processed' in data.columns]

    if args.all:
        print("Creating dataset with ALL utterances")
    else:
        print("Creating dataset with FIRST utterances")

    tqdm.pandas()
    data["TEXT"] = data.yt_processed.apply(lambda x: x.replace("\n", " ").replace("\t", " ").split())
    data["NER_TAGS"] = data.progress_apply(make_taglist, args=(ent_names, args.baseline_names, args.all), axis=1)
    
    data.to_parquet(args.output)

def parse_args():
    parser = argparse.ArgumentParser(description='Make BIO tag list for NER task.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output parquet file.')
    parser.add_argument('--baseline_names', action='store_true', help='Whether to change entity class name to the ones used in the baseline approach.')
    parser.add_argument('--all', action='store_true', help='Whether to find all or only the first occurance per entity in the string.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()