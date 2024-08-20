from pydantic import BaseModel
from typing import List
from enum import Enum


class Label(str, Enum):
    title = 'title'
    performer = 'performer'
    other = 'other'

class MusicEntity(BaseModel):
    """Data model of a music entity"""
    utterance: str 
    label: str
    cue: str

    class Config:  
        use_enum_values = True
        
class EntityList(BaseModel):
    """Data model for list of music entities."""
    content: List[MusicEntity]
    
class Example(BaseModel):
    """
    Data model for a few-shot example.
    """
    text: str
    output: EntityList
