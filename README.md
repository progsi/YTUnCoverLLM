# YTUnCoverLLM
An approach to process YouTube metadata to detect songs that are likely contained in videos.

## Song Entities in Online Videos

The relationship of online videos and songs is an n-to-m relationship. We want to use LLMs to extract song-level information from online video metadata. A performed song can have the following attributes:
- original song title
- cover song title
- original artist(s)
- covering artist(s)
- original album (rare)
- cover album (rare)
- venue (eg. "Luna Park")
- instruments (instrumental covers, eg "guitar cover")
- genres (cross-version covers, eg. "metal cover")
- sections (eg. "solo", "intro")

In the situational context of an online video, other information could be interesting:
- person (eg. tv show hosts)
- city (also interesting for live concerts)
- movies (sometimes songs accompany movie trailers...)
- shows (...or are performed in tv shows)
- ...