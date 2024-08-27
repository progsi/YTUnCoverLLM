import argparse
from pydantic import BaseModel
import numpy as np
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
For the following, return only the answer (or the comma-separated answers) as a String without further explanations:

Who is the original performing artist for the song "{title}" released in the year {year}? 
"""
prompt_template = PromptTemplate(prompt_str)

def get_original(row):

    def get_year(row):
        if row.release_year:
            year = row.release_year
        elif row.first_perf_year:
            year = row.first_perf_year
        elif row.first_year:
            year = row.first_year
        else:
            year = row.other_release_year
        return year

    if type == "Original":
        title = row.perf_title
        performer = row.performer
        year = get_year(row)

    else:
        title = row.work_title if row.work_title is not None else row.perf_title 
        if row.third_artist:
            performer = row.third_artist 
        elif row.second_artist:
            performer = row.second_artist
        else:
            performer = row.first_artist # artist of perf
        year = get_year(row)
    composer = row.written_by
    return {
        "perf_id": row.perf_id,
        "work_id": int(row.work_id) if not np.isnan(row.work_id) else None,
        "title": title,
        "performer": performer,
        "year": int(year) if not np.isnan(year) else None,
        "composer": composer
    }


def main() -> None:
    args = parse_args()

    predict_kwargs = {}

    llm = Ollama(model=args.llm, temperature=0.0, request_timeout=80.0)

    data = pd.read_json(args.input, lines=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        for i, row in tqdm(data.iterrows(), total=len(data)):
            
            original = get_original(row)
            true_title = original.get("title")
            true_performer = original.get("performer")
            true_year = original.get("year")
            true_composer = original.get("composer")

            # extract with LLM
            predict_kwargs["title"] = true_title
            predict_kwargs["year"] = true_year
            
            # define output
            output = {}
            output["work_id"] = original.get("work_id")
            output["perf_id"] = original.get("perf_id")
            output["true_title"] = true_title
            output["true_performer"] = true_performer
            output["true_composer"] = true_composer
            output["true_year"] = true_year

            try:
                resp = llm.complete(prompt_template.format(title=true_title, year=true_year))
            except (ValidationError, ValueError) as e:
                print(f"Exception {e} for text: {true_title}")
                artists = []
            output["pred_performer"] = resp.text

            line = json.dumps(output, ensure_ascii=False)
            f.write(line + '\n')


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run named LLM-based information extraction using pydantic output schema.')
    parser.add_argument('--llm', type=str, help='large language model to use.')
    parser.add_argument('-i', '--input', type=str, help='Path of grouped SHS100k2 file.', default="data/raw/shs100k_metadata.jsonl")
    parser.add_argument('-o', '--output', type=str, help='Output path.')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()