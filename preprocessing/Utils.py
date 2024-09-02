import pandas as pd
import unicodedata
import re
from typing import List, Dict, Tuple, Union
from unidecode import unidecode
import numpy as np
from rapidfuzz.fuzz import partial_ratio_alignment
from itertools import combinations


# order important! Due to overlaps between title and artist strings.
SONG_ATTRS = ["title", "title_perf", "title_work", "performer", "performer_perf", "performer_work"]
CLASS_ATTRS = ["split", "set_id", "ver_id", "yt_id"]
YT_ATTRS = ["video_title", "description"]

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
O_LABEL = "O"

APOSTROPHES = "'’‘`´ʻʼʽ"

def replace_linebreaks_tabs(s: str) -> str:
    return s.replace("\n", " ").replace("\t", " ").replace("\r", " ")

def remove_bracket_with_one_content(s: str) -> str:
    """Remove brackets "[CONTENT]" with their content if its one token only. 
    Common in SHS metadata to denote nationality of the performer.
    Args:
        s (str): string input
    Returns:
        str: processed string
    """
    pattern = r'\(\w+\)|\[\w+\]|\{\w+\}'
    return re.sub(pattern, '', s) 

def remove_brackets_and_all_content(s):
    """Remove brackets "[CONTENT CONTENT]" with their content. 
    Args:
        s (str): string input
    Returns:
        str: processed string
    """
    pattern = r'\[.*?\]|\(.*?\)|\{.*?\}'
    return re.sub(pattern, '', s)

def remove_bracket_only(s: str) -> str:
    pattern = r'[\[\]\(\)\{\}]'
    return re.sub(pattern, '', s)

def unicode_normalize(s: str) -> str:
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')

def isolate_special_chars(s: str, exclude: str = "'") -> str:
    """Separates special chars.
    Args:
        s (str): input string
        exclude (str): string of chars to not isolate (typically ')
    Returns:
        str: string with isolated special chars
    """
    special_chars = r'([!\"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~])'.replace(exclude, "")
    s = re.sub(special_chars, r' \1 ', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

def only_special_chars(s: str) -> bool:
    pattern = r'^[^\w\s]+$'
    return bool(re.match(pattern, s))

def only_underscore(s: str) -> bool:
    chars = set(s)
    return len(chars) == 1 and '_' in chars

def remove_attached_special_chars(s: str) -> str:
    """Remove only the attached special chars for more robust matching but still keeping initial 
    word indices.
    Args:
        s (str): 
    Returns:
        str: 
    """
    pattern = r'[^a-zA-Z0-9 ]'
    def remove_attached(w: str) -> str:
        if not only_special_chars(w) and not only_underscore(w):
            cleaned = re.sub(pattern, '', w)
            if len(cleaned.strip()) > 0:
                return cleaned
        return w
    s = ' '.join([remove_attached(w) for w in s.split()])
    return s

def unidecode_letters(s: str) -> str:
    def replace_with_unidecode(match):
        char = match.group(0)
        return unidecode(char)
    s = re.sub(r'[^\W\d_]', replace_with_unidecode, s)
    return s

def simplify_string(s: str) -> str:
    """Only retain space, latin chars and numbers. Remove attached special chars
    Args:
        s (str): 
    Returns:
        str: 
    """
    # remove apostrophe
    for a in APOSTROPHES:
        s = s.replace(a, "")

    # to basic latin 
    s = ' '.join([unidecode_letters(w).replace(" ", "") for w in s.split()])
    # isolation
    s = isolate_special_chars(s)
    return s

def basic_preprocessing(texts: List[str]) -> List[str]:
    """Basic preprocessing pipeline only doing lowercase and removing newlines etc.
    Args:
        texts (List[str]): 
    Returns:
        List[str]: processed textes
    """
    return [replace_linebreaks_tabs(s.lower().replace('"', "'")) for s in texts]

def is_one_special_char(char):
    special_char_regex = re.compile(r'[^a-zA-Z0-9\s]')
    return len(char) == 1 and bool(special_char_regex.search(char))

def strip_list_special_chars(l: List[str]) -> List[str]:
    """Strip list with strings from special chars at the beginning and end.
    Args:
        l (List[str]): original list
    Returns:
        List[str]: stripped list
    """
    if is_one_special_char(l[0]):
        l = l[1:]
    if len(l) > 0 and is_one_special_char(l[-1]):
        l = l[:-1]
    return l

def char_idx_to_word_idx(s: str, idx: int) -> int:
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
        cur_len = len(word)
        if cur_idx <= idx < cur_idx + cur_len:
            return w_idx
        cur_idx += cur_len + 1
    return 

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

def find_sublist_indices(superlist: np.ndarray, sublist: np.ndarray) -> List[int]:
    """
    Find all indices where the sublist occurs in the array of text_list.

    Args:
        text_list (np.ndarray): The array of words to search within.
        sublist (np.ndarray): The sublist of words to search for.

    Returns:
        List[int]: A list of starting indices where the sublist occurs in the array of words.
    """
    indices = []
    sublist_len = len(sublist)
    
    for i in range(len(superlist) - sublist_len + 1):
        if np.array_equal(superlist[i:i + sublist_len], sublist):
            indices.append(i)
    
    return indices


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
    
def find_ent_utterance(text: str, start_idx: int, end_idx: int) -> List[str]:
    
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


def make_taglist(item: Union[pd.Series, dict], ent_names: List[str], baseline_name: bool, all: bool, 
                min_r: int, text_col: str = "yt_processed") -> List[str]:
    """Creates a tag list with IOB tags for NER based on yt metadata (eg. yt_processed) in the dataframe item.
    Args:
        item (Union[pd.Series, dict]): Row in the dataframe.
        ent_names (List[str]): list of entity names
        baseline_name (bool): whether to change entity names to coarse attributes from the baseline approach
        all (bool): Whether to search for all occurances or only the first.
        min_r (int): minimum ratio for partial ratio. If none, exact matching.
    Returns:
        List[str]: list with IOB tags
    """
    text = item[text_col]
    # simplify for more robust matching
    match_text = simplify_string(text)

    ent_spans = {}
    for ent_name in ent_names:
        # assume list (split performers), otherwise, make one element list
        if type(item) == pd.Series:
            ents = item[("shs_processed", ent_name)]
        else:
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

def write_biotag(data: pd.DataFrame, filepath: str, IOB_col : str):
    """Writes a dataframe to NER IOB tag format. From:
    https://stackoverflow.com/questions/67200114/convert-csv-data-into-conll-bio-format-for-ner
    Args:
        data (pd.DataFrame): the dataframe.
        filepath (str): the output filepath.
        IOB_col (str): name of IOB column
    """
    with open(filepath, "w") as f_out:
        for _, line in data.iterrows():
            for txt, tag in zip(line["TEXT"], line[IOB_col]):
                print("{}\t{}".format(txt, tag), file=f_out)
            print(file=f_out)