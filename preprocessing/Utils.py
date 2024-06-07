import pandas as pd
import unicodedata
import re
from typing import List
from unidecode import unidecode
import numpy as np

# order important! Due to overlaps between title and artist strings.
SONG_ATTRS = ["title", "title_perf", "title_work", "performer", "performer_perf", "performer_work"]
CLASS_ATTRS = ["split", "set_id", "ver_id", "yt_id"]
YT_ATTRS = ["video_title", "channel_name", "description"]

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
    return [replace_linebreaks_tabs(s.lower()) for s in texts]

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
