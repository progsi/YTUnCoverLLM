import pandas as pd
import unicodedata
import re

def replace_linebreaks_tabs(series: pd.Series) -> pd.Series:
    return series.str.replace("\n", " ").str.replace("\t", " ").str.replace("\r", " ")

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



