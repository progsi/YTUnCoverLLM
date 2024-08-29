import argparse
import numpy as np
import pandas as pd
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from pydantic_core._pydantic_core import ValidationError
from tqdm import tqdm
from src.Prompts import Q1, Q2, Q3
import os 
import json
     
instruction_str = """\
You are asked a question for which the answer is one or more person(s) or music groups. 
Please only reply by the correct answer and separate the person(s) or groups by commas if there are multiple correct ones.

{question}
"""

prompt_template = PromptTemplate(instruction_str)

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

    llm = Ollama(model=args.llm, temperature=0.0, request_timeout=600.0)

    def pred_llm(question) -> str:
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
            output["work_id"] = original.get("work_id")
            output["perf_id"] = original.get("perf_id")
            output["title_perf"] = ptitle
            output["artist_original"] = operformer
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
                question=Q1.format(title_original=otitle, year_original=oyear))
            # Q2 --> performer by year and release type?
            output["AW2"] = pred_llm(
                question=Q2.format(title_perf=ptitle, release_type=get_release_type_str(row.release_type), year_perf=pyear))
            # Q3 --> composer/writer?
            output["AW3"] = pred_llm(
                question=Q3.format(title_perf=ptitle, artist_perf=pperformer, year_perf=pyear))

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