import pandas as pd

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

# for preprocessing
ARTICLES = [
    # french
    "le", "les", 
    # spanish
    "la", "el", "los", "las", 
    # german
    "der", "die", "das", 
    # portuguese
    "a", "o", "os", "as"
    # italian
    "il", "lo", "l'", "i", "gli", "le",
    # english
    "the"]

AND_VARIATIONS_LONG = [
# and + genetive
"and his", "and her", "y su", "e la sua", 
"e la seu", "e seu", "e sua", "und sein", "und ihr", 
"und seine", "und ihre", "et le", "et son", "et ses", 
"et les", 
# with...
"with her", "with his", "with the", 
"mit ihrem", "mit ihren", "mit seinem", "mit seinen",
"com o seu", "com o"]
AND_VARIATIONS_SHORT = ["&", "and", "y", "e", "et", "und", ",", "-", "con", "avec", "mit", "com", "with"]

def replace_linebreaks_tabs(series: pd.Series) -> pd.Series:
    return series.str.replace("\n", " ").str.replace("\t", " ").str.replace("\r", " ")

def split_performers_series(performers: pd.Series, featuring_token: str = "featuring") -> pd.Series:
    """Splits the raw performer string by handcrafted criteria.
    Args:
        performers (pd.Series): Series of lists where each element in the list is a single performer.
    Returns:
        pd.Series:
    """
    # lowercase
    performers = performers.str.lower()
    # normalize punctiation
    performers = performers.str.replace(" feat. ", " feat ").str.replace(" ft. ", " ft ")
    # normalize featuring abbrv.
    performers = performers.str.replace(" feat ", f" {featuring_token} ").str.replace(" ft ", f" {featuring_token} ")

    performers = performers.str.replace(", ", " , ")

    # replace long variations
    for and_var in AND_VARIATIONS_LONG:
        performers = performers.str.replace(f" {and_var} ", f" {featuring_token} ")

    # replace short variations
    for and_var in AND_VARIATIONS_SHORT:
        performers = performers.str.replace(f" {and_var} ", f" {featuring_token} ")

    def only_space(s):
        """Check if string contains only spaces, tabs, newlines.
        Args:
            s (str): input string
        Returns:
            bool: 
        """
        return all(char.isspace() for char in s)
    
    return performers.apply(lambda x: [t.strip() for t in x.split("featuring") if not only_space(t)])
