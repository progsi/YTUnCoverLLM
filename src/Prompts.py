

PROMPT_ZEROSHOT_V3 = """\
From the following text which contains a user requests for music suggestions, extract all the relevant entities that you find. 
Entity Attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'title' (if the utterance refers is a song or album name), 'performer' (if utterance is a performing artist) or 'Other' for any other entity type. 
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here is the text: {text}
"""

PROMPT_FEWSHOT_V3 =  """\
From the following text which contains a user requests for music suggestions, extract all the relevant entities that you find. 
Entity Attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'title' (if the utterance refers is a song or album name), 'performer' (if utterance is a performing artist) or 'Other' for any other entity type. 
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here are {k} examples: 
{few_shot_examples}
    
Here is the text: {text}
"""

PROMPT_ZEROSHOT_V3_OUTPUT = """\
From the following text which contains a user requests for music suggestions, extract all the relevant entities that you find. 
Entity Attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'title' (if the utterance refers is a song or album name), 'performer' (if utterance is a performing artist) or 'Other' for any other entity type. 
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")

Output Format:
    The predefined schema without additional text. Empty, if no entity is contained.

Here is the text: {text}
"""

PROMPT_FEWSHOT_V3_OUTPUT = """\
From the following text which contains a user requests for music suggestions, extract all the relevant entities that you find. 
Entity Attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'title' (if the utterance refers is a song or album name), 'performer' (if utterance is a performing artist) or 'Other' for any other entity type. 
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")

Output Format:
    The predefined schema without additional text. Empty, if no entity is contained. contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")

Here are {k} examples: 
{few_shot_examples}
    
Here is the text: {text}
"""



PROMPT_ZEROSHOT_V2 = """\
From the following text which contains a user requests for music suggestions, extract all the music entities (songs, albums, performing artists) that you find. 
A music entity has the following attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'title' (eg. a song title, an album title, a symphony) or it can be 'performer' which refers to a performing musical artist. It cannot refer to more general concepts like genres, moods, instruments or other musical characteristics.
    - cue: The contextual cue which indicates the musical entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here is the text: {text}
"""

PROMPT_FEWSHOT_V2 =  """\
From the following text which contains a user requests for music suggestions, extract all the music entities (songs, albums, performing artists) that you find.
A music entity has the following attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'title' (eg. a song title, an album title, a symphony) or it can be 'performer' which refers to a performing musical artist. It cannot refer to more general concepts like genres, moods, instruments or other musical characteristics.
    - cue: The contextual cue which indicates the musical entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")

Here are {k} examples: 
{few_shot_examples}
    
Here is the text: {text}
"""



PROMPT_ZEROSHOT_V1 = """\
From the following text which contains a user requests for music suggestions, extract all the music entities.
A music entity has the following attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles".
    - label: The label of the entity. It can either be 'title' (eg. a song title, an album title, a symphony) or it can be 'performer' which refers to a performing musical artist.
    - cue: The contextual cue which indicates the musical entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here is the text: {text}
"""

PROMPT_FEWSHOT_V1 = """\
From the following text which contains a user requests for music suggestions, extract all the music entities.
A music entity has the following attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles".
    - label: The label of the entity. It can either be 'title' (eg. a song title, an album title, a symphony) or it can be 'performer' which refers to a performing musical artist.
    - cue: The contextual cue which indicates the musical entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")

Here are {k} examples: 
{few_shot_examples}
    
Here is the text: {text}
"""
