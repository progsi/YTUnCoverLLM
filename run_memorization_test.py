import argparse
import numpy as np
import pandas as pd
from typing import Union
from src.Schema import MemorizationAW
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.program import LLMTextCompletionProgram, FunctionCallingProgram
from pydantic_core._pydantic_core import ValidationError
from tqdm import tqdm
from src.Prompts import Q1, Q2, Q3
from src.Utils import get_key
import os 
import json
     
instruction_str = """\
You are asked a question for which the answer is one or more person(s) or music groups. 
Please only reply by the correct answer without additional text and separate the person(s) or groups by commas if there are multiple correct ones.

{question}
"""

prompt_template = PromptTemplate(instruction_str)

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
        "output_cls": MemorizationAW,
        "allow_multiple": False,
        "verbose": False,
    }
    instruction_str = """\
    You are asked a question for which the answer is one or more person(s) or music groups. 
    Please only reply by the correct answer and no additional text. Separate the person(s) or groups by commas if there are multiple correct ones.

    {question}
    """
    kwargs["prompt_template_str"] = instruction_str
 
    if is_openai:
        llm = OpenAI(model=model, api_key=get_key("openai"), temperature=0.0)
        kwargs["llm"] = llm
        program = OpenAIPydanticProgram.from_defaults(**kwargs)
        print(f"{model} loaded successfully via OpenAI API.")
    else:
        try:
            llm = Ollama(model=model, temperature=0.0, is_function_calling_model=True)
            kwargs["llm"] = llm

            try:
                program = FunctionCallingProgram.from_defaults(**kwargs)
            except:
                print(f"{model} does not support function calling.")
                program = LLMTextCompletionProgram.from_defaults(**kwargs)

            print(f"{model} loaded via Ollama.")
        except:
            print(f"{model} appears to be not available on Ollama!")

    return program

def get_original(row):
  
    if type == "Original":
        title = row.perf_title
        performer = row.performer
        if row.release_year:
            year = row.release_year
        elif row.performed_in_year:
            year = row.performed_in_year
        elif row.performed_live_in_year:
            year = row.performed_live_in_year
        elif row.first_perf_year:
            year = row.first_perf_year
        elif row.first_year:
            year = row.first_year
        else:
            year = row.other_release_year
    else:

        def get_original_ver(row):
            if row.third_artist and row.third_year:
                return row.third_artist, row.third_year
            elif row.second_artist and row.second_year:
                return row.second_artist, row.second_year
            elif row.first_artist and row.first_year:
                return row.first_artist, row.first_year
            elif row.perf_title and row.first_perf_year:
                return row.perf_title, row.first_perf_year
            else:
                return row.perf_title, row.release_year

        title = row.work_title if row.work_title is not None else row.perf_title 
        performer, year = get_original_ver(row)

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

    as_program = args.as_program or args.openai
    if as_program:
        llm = init(args.llm, args.openai)
    else:
        llm = Ollama(model=args.llm, temperature=0.0, request_timeout=600.0)

    def pred_llm(question, as_program) -> str:
        if as_program:
            try:
                kwargs = {"question": question} 
                aw = llm(**kwargs)
                return ','.join([s.replace(",", " ") for s in aw.dict()["names"]])
            except Exception as e:
                print(f"Exception {e} for question: {question}")
        else:
            try:
                return llm.complete(prompt_template.format(question=question)).text
            except Exception as e:
                print(f"Exception {e} for question: {question}")
                return ''

    data = pd.read_json(args.input, lines=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        for i, row in tqdm(data.iterrows(), total=len(data)):
            
            original = get_original(row)
            otitle = original.get("title")
            operformer = original.get("performer")
            oyear = original.get("year")
            oyear = int(oyear) if oyear else "(year not known)"
            composer = original.get("composer")
            ptitle = row.perf_title
            pperformer = row.perf_artist
            pyear = int(row.release_year) if not np.isnan(row.release_year) else  "(year not known)"

            # extract with LLM
            predict_kwargs["title_perf"] = otitle
            predict_kwargs["year_original"] = oyear
            
            # define output
            output = {}
            output["set_id"] = row.set_id
            output["work_id"] = original.get("work_id")
            output["perf_id"] = original.get("perf_id")
            output["title_original"] = otitle
            output["artist_original"] = operformer
            output["title_perf"] = ptitle
            output["artist_perf"] = operformer
            output["composer"] = composer
            output["year_original"] = oyear

            def get_release_type_str(release_type):
                if release_type:
                    if release_type[0].lower() in "aeiou":
                        return f"an {release_type}"
                    else:
                        return f"a {release_type}"
                return f"a release"

            # Q1 --> original performer?
            output["AW1"] = pred_llm(
                question=Q1.format(title_original=otitle, year_original=oyear), as_program=as_program)
            # Q2 --> performer by year and release type?
            output["AW2"] = pred_llm(
                question=Q2.format(title_perf=ptitle, release_type=get_release_type_str(row.release_type), 
                                   year_perf=pyear), as_program=as_program)
            # Q3 --> composer/writer?
            output["AW3"] = pred_llm(
                question=Q3.format(title_perf=ptitle, artist_perf=pperformer, year_perf=pyear), 
                as_program=as_program)

            line = json.dumps(output, ensure_ascii=False)
            f.write(line + '\n')


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run named LLM-based information extraction using pydantic output schema.')
    parser.add_argument('--llm', type=str, help='large language model to use.')
    parser.add_argument('-i', '--input', type=str, help='Path of SHS100k metadata file.', default="data/raw/shs100k_metadata.jsonl")
    parser.add_argument('-o', '--output', type=str, help='Output path.')
    parser.add_argument("--as_program", action="store_true", help="Use llamaindex program class") 
    parser.add_argument("--openai", action="store_true", help="Use OpenAI API") 

    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()