import argparse
from pydantic import BaseModel
import pandas as pd
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from src.Utils import get_key
from typing import List, Union
from llama_index.core import PromptTemplate
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.program import LLMTextCompletionProgram, FunctionCallingProgram
from pydantic_core._pydantic_core import ValidationError
from tqdm import tqdm
from src.Utils import read_IOB_file, transform_to_dict, write_jsonlines
import os 
import json

prompt_str = """\
For the following song or medley, please return the original performing artist(s) and some notable covering artist(s). 
Please output into a JSON with keys 'original' and 'covering' mapping to the respective lists of strings. 
The title is: "{title}"
"""
prompt_template = PromptTemplate(prompt_str)

def main() -> None:
    args = parse_args()

    predict_kwargs = {}

    llm = Ollama(model=args.llm, temperature=0.0, json_mode=True, request_timeout=80.0)

    data = pd.read_json(args.input, lines=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        for i, row in tqdm(data.iterrows(), total=len(data)):

            set_id = row.set_id
            true_title = row.title
            true_performer = row.performer

            # extract with LLM
            predict_kwargs["title"] = true_title
            
            # define output
            output = {}
            output["set_id"] = set_id
            output["true_title"] = true_title
            output["true_performer"] = true_performer

            try:
                resp = llm.complete(prompt_template.format(title=true_title))
            except (ValidationError, ValueError) as e:
                print(f"Exception {e} for text: {true_title}")
                artists = []
            output["extracted"] = json.loads(resp.text)

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