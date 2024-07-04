import argparse
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from src.Utils import get_key
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Union
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.program import LLMTextCompletionProgram
from pydantic_core._pydantic_core import ValidationError
from tqdm import tqdm
from src.Utils import read_IOB_file, transform_to_dict, write_jsonlines
import os 
import json


prompt_template = """\
From the following text which contains a user requests for music suggestions, extract all the music entities.
A music entity has the following attributes:
    - utterance: The utterance of the entity in the text. For example "the beatles" in "recommend me music like the beatles".
    - label: The label of the entity. It can either be 'title' (eg. a song title, an album title, a symphony) or it can be 'performer' which refers to a performing musical artist.
    - cue: The contextual cue which indicates the musical entity (eg. "music like" in "recommend me music like the beatles" indicating "the beatles")
Here is the text: {text}
"""

class Label(str, Enum):
    title = 'title'
    performer = 'performer'

class MusicEntity(BaseModel):
    """Data model of a music entity"""
    utterance: str 
    label: Label
    cue: str

    class Config:  
        use_enum_values = True
        
class EntityList(BaseModel):
    """Data model for list of music entities."""
    content: List[MusicEntity]
    

def init(model: str) -> Union[OpenAIPydanticProgram, LLMTextCompletionProgram]:
    """Load the program for structured output based on the LLM used
    Args:
        model (str): LLM name string
    Returns:
        Union[OpenAIPydanticProgram, LLMTextCompletionProgram]: program module from Llamaindex
    """
    try:
        llm = OpenAI(model=model, api_key=get_key("openai"), temperature=0.0)
        program = OpenAIPydanticProgram.from_defaults(
            output_cls=EntityList,
            llm=llm,
            prompt_template_str=prompt_template,
            allow_multiple=False,
            verbose=False,
        )
        print(f"{model} loaded successfully via OpenAI API.")
    except:
        llm = Ollama(model=model, temperature=0.0)
        program = LLMTextCompletionProgram.from_defaults(
        output_cls=EntityList,
        llm=llm,
        prompt_template_str=prompt_template,
        allow_multiple=False,
        verbose=False,
        )
        print(f"{model} loaded via Ollama.")
    return program


def main() -> None:
    args = parse_args()

    program = init(args.llm)
    
    texts, labels = read_IOB_file(args.input)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        for tokens, iob in tqdm(zip(texts, labels)):

            text = ' '.join(tokens)
            true_ents = transform_to_dict(tokens, iob)

            # put input data and true entities
            output = {}
            output["text"] = text
            output["performers"] = true_ents.get("Artist") or []
            output["titles"] = true_ents.get("WoA") or []

            # extract with LLM
            try:
                ent_list = program(text=text)
                llm_ents = [ent.model_dump() for ent in ent_list.content]
            except (ValidationError, ValueError) as e:
                print(f"Exception {e} for text: {text}")
                llm_ents = []
            output["extracted"] = llm_ents

            line = json.dumps(output, ensure_ascii=False)
            f.write(line + '\n')


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run named LLM-based information extraction using pydantic output schema.')
    parser.add_argument('--llm', type=str, help='large language model to use.', default="gpt-3.5-turbo-0125")
    parser.add_argument('-i', '--input', type=str, help='Dataset path.')
    parser.add_argument('-o', '--output', type=str, help='Output path.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()