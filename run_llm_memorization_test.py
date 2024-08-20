import argparse
from pydantic import BaseModel
import pandas as pd
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from src.Utils import get_key
from typing import List, Union
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.program import LLMTextCompletionProgram, FunctionCallingProgram
from pydantic_core._pydantic_core import ValidationError
from tqdm import tqdm
from src.Utils import read_IOB_file, transform_to_dict, write_jsonlines
import os 
import json

class ArtistsV1(BaseModel):
    """Data model for list of music entities."""
    original: str
    covering: List[str]

class ArtistsV2(BaseModel):
    """Data model for list of music entities."""
    original: List[str]
    covering: List[str]

class Artist(BaseModel):
    """Data model for list of music entities."""
    original: List[str]
    covering: List[str]

PROMPT = """\
For the following {type}, please return the original performing artist(s) and some notable covering artist(s). 
Please output in the defined output schema. The title of the {type} is: "{title}"
"""

OPEN_AI_MODELS = ["gpt-3.5", "gpt-4"]

def init(model: str, is_openai: bool = True) -> Union[OpenAIPydanticProgram, LLMTextCompletionProgram]:
    """Load the program for structured output based on the LLM used
    Args:
        model (str): LLM name string
        few_shot_set (FewShotSet): 
        sampling_method (str): sampling method for k examples. Defaults to random = "rand"
        is_openai (bool):
    Returns:
        Union[OpenAIPydanticProgram, LLMTextCompletionProgram]: program module from Llamaindex
    """
    # set kwargs
    kwargs = {
        "output_cls": ArtistsV1,
        "allow_multiple": False,
        "verbose": False,
    }
    kwargs["prompt_template_str"] = PROMPT
 
    if is_openai:
        llm = OpenAI(model=model, api_key=get_key("openai"), temperature=0.0)
        kwargs["llm"] = llm
        program = OpenAIPydanticProgram.from_defaults(**kwargs)
        print(f"{model} loaded successfully via OpenAI API.")
    else:
        try:
            llm = Ollama(model=model, temperature=0.0)
            kwargs["llm"] = llm.as_structured_llm(ArtistsV2)
            # program = LLMTextCompletionProgram.from_defaults(**kwargs)
            program = FunctionCallingProgram.from_defaults(**kwargs)
            print(f"{model} loaded via Ollama.")
        except:
            print(f"{model} appears to be not available on Ollama!")

    return program


def main() -> None:
    args = parse_args()

    is_openai = any([llm_name in args.llm for llm_name in OPEN_AI_MODELS])

    predict_kwargs = {}

    program = init(args.llm, is_openai)
    
    data = pd.read_json(args.input, lines=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        for i, row in data.iterrows():

            set_id = row.set_id
            true_title = row.title
            true_performer = row.performer

            # extract with LLM
            predict_kwargs["title"] = true_title
            predict_kwargs["type"] = "medley" if "/" in true_title else "song"
            
            # define output
            output = {}
            output["set_id"] = set_id
            output["true_title"] = true_title
            output["true_performer"] = true_performer

            try:
                artists = program(**predict_kwargs)
                artists = artists.json()
            except (ValidationError, ValueError) as e:
                print(f"Exception {e} for text: {true_title}")
                artists = []
            output["extracted"] = artists

            line = json.dumps(output, ensure_ascii=False)
            f.write(line + '\n')


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run named LLM-based information extraction using pydantic output schema.')
    parser.add_argument('--llm', type=str, help='large language model to use.')
    parser.add_argument('-i', '--input', type=str, help='Path of grouped SHS100k2 file.', default="data/raw/shs100k2_grouped.jsonl")
    parser.add_argument('-o', '--output', type=str, help='Output path.')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()