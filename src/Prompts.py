PROMPT_ZEROSHOT = """\
From the following text which contains a user requests for music suggestions, extract all the music entities.
A music entity has the following attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles".
    - label: The label of the entity. It can either be 'title' (eg. a song title, an album title, a symphony) or it can be 'performer' which refers to a performing musical artist.
    - cue: The contextual cue which indicates the musical entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here is the text: {text}
"""

PROMPT_FEWSHOT = """\
From the following text which contains a user requests for music suggestions, extract all the music entities.
A music entity has the following attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles".
    - label: The label of the entity. It can either be 'title' (eg. a song title, an album title, a symphony) or it can be 'performer' which refers to a performing musical artist.
    - cue: The contextual cue which indicates the musical entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")

Here are {k} examples: 
{few_shot_examples}
    
Here is the text: {text}
"""
