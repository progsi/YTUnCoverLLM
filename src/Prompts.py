
Q1 = """\
    Who is or who are the original performing artist(s) of the song "{title_original}" released in the year {year_original}?
    """

Q2 = """\
    Who performed the song "{title_perf}" on {release_type} in the year {year_perf}?
"""

Q3 = """\
    Who wrote the original song of the cover version "{title_perf}" performed by {artist_perf} in the year {year_perf}?
"""

PROMPT_ZEROSHOT_V4_OUTPUT = """\
From the following text of user-generated content from the web, extract relevant music entities that you find. Return a JSON with a key "entities" which maps to one JSON per entity you find. Each entity has the following keys mapping to the respective values:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'TITLE' (if the utterance refers is a song or album name), 'PERFORMER' (if utterance is a performing artist) or 'OTHER' for any other entity type. 
    - normalized: The actual official name of the entity utterated, but with corrected spelling and non-abbreviated. If the utterance is correct, the utterance should be copied here.
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")

Output Format:
    The predefined schema without additional text. If you do not find an entity, output an empty instance of that schema.

Here is the text: {text}
"""

PROMPT_ZEROSHOT_V4_JSON = """\
From the following text of user-generated content from the web, extract relevant music entities that you find. Return a JSON with a key "entities" which maps to one JSON per entity you find. Each entity has the following keys mapping to the respective values:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'TITLE' (if the utterance refers is a song or album name), 'PERFORMER' (if utterance is a performing artist) or 'OTHER' for any other entity type. 
    - normalized: The actual official name of the entity utterated, but with corrected spelling and non-abbreviated. If the utterance is correct, the utterance should be copied here.
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here is the text: {text}
"""

PROMPT_FEWSHOT_V4_JSON =  """\
From the following text of user-generated content from the web, extract relevant music entities that you find. Return a JSON with a key "entities" which maps to one JSON per entity you find. Each entity has the following keys mapping to the respective values:
Entity Attributes:
    - utterance: The utterance of the entity exactly as in the text. For example "tha beatles" in "recommend me music like tha beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'TITLE' (if the utterance refers is a song or album name), 'PERFORMER' (if utterance is a performing artist) or 'OTHER' for any other entity type. 
    - normalized: The actual official name of the entity utterated, but with corrected spelling and non-abbreviated (eg. "the beatles" for the above example). If the utterance is correct, the utterance should be copied here. 
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here are {k} examples: 
{few_shot_examples}
    
TEXT: {text}
RESPONSE: 
"""

PROMPT_ZEROSHOT_V4 = """\
From the following text of user-generated content from the web, extract relevant music entities that you find.  
Entity Attributes:
    - utterance: The utterance of the entity exactly as in the text. For example "tha beatles" in "recommend me music like tha beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'TITLE' (if the utterance refers is a song or album name), 'PERFORMER' (if utterance is a performing artist) or 'OTHER' for any other entity type. 
    - normalized: The actual official name of the entity utterated, but with corrected spelling and non-abbreviated (eg. "the beatles" for the above example). If the utterance is correct, the utterance should be copied here. 
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here is the text: {text}
"""

PROMPT_FEWSHOT_V4 =  """\
From the following text of user-generated content from the web, extract relevant music entities that you find.  
Entity Attributes:
    - utterance: The utterance of the entity exactly as in the text. For example "tha beatles" in "recommend me music like tha beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'TITLE' (if the utterance refers is a song or album name), 'PERFORMER' (if utterance is a performing artist) or 'OTHER' for any other entity type. 
    - normalized: The actual official name of the entity utterated, but with corrected spelling and non-abbreviated (eg. "the beatles" for the above example). If the utterance is correct, the utterance should be copied here. 
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here are {k} examples: 
{few_shot_examples}
    
Here is the text: {text}
"""

PROMPT_ZEROSHOT_V3 = """\
From the following text which contains a user requests for music suggestions, extract all the relevant entities that you find. 
Entity Attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'TITLE' (if the utterance refers is a song or album name), 'PERFORMER' (if utterance is a performing artist) or 'OTHER' for any other entity type. 
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here is the text: {text}
"""

PROMPT_FEWSHOT_V3 =  """\
From the following text which contains a user requests for music suggestions, extract all the relevant entities that you find. 
Entity Attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'TITLE' (if the utterance refers is a song or album name), 'PERFORMER' (if utterance is a performing artist) or 'OTHER' for any other entity type. 
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here are {k} examples: 
{few_shot_examples}
    
Here is the text: {text}
"""

PROMPT_ZEROSHOT_V3_OUTPUT = """\
From the following text which contains a user requests for music suggestions, extract all the relevant entities that you find. 
Entity Attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'TITLE' (if the utterance refers is a song or album name), 'PERFORMER' (if utterance is a performing artist) or 'OTHER' for any other entity type. 
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")

Output Format:
    The predefined schema without additional text. If you do not find an entity, output an empty instance of that schema.

Here is the text: {text}
"""

PROMPT_FEWSHOT_V3_OUTPUT = """\
From the following text which contains a user requests for music suggestions, extract all the relevant entities that you find. 
Entity Attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles". An utterance can only be of types for which labels are defined.
    - label: The label of the entity. It can either be 'TITLE' (if the utterance refers is a song or album name), 'PERFORMER' (if utterance is a performing artist) or 'OTHER' for any other entity type. 
    - cue: The contextual cue which indicates the entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")

Output Format:
    The predefined schema without additional text. If you do not find an entity, output an empty instance of that schema.

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
