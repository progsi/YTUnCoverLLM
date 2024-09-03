import argparse
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from src.Utils import get_key
from src.Schema import EntityListV2
from src.Prompts import PROMPT_ZEROSHOT_V4_JSON, PROMPT_FEWSHOT_V4
from src.FewShot import FewShotSet
from typing import List, Union
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.program import FunctionCallingProgram
from pydantic_core._pydantic_core import ValidationError
from tqdm import tqdm
from src.Utils import read_IOB_file, transform_to_dict, write_jsonlines
import os 
import json
import pandas as pd

def main() -> None:
    args = parse_args()
    k = args.nexamples
    bzeroshot = k > 0

    if bzeroshot:
        few_shot_set = None
        predict_kwargs = {}
    else:
        few_shot_set = FewShotSet(args.input, False)
        print(f"Few-Shot set successfully loaded; k={k}")
        predict_kwargs = {"k": k}

    prompt_template = PROMPT_ZEROSHOT_V4_JSON if bzeroshot else few_shot_set.get_prompt_template(args.sampling_method)
    
    llm = Ollama(model=args.llm, temperature=0.0, json_mode=True, request_timeout=600.0)

    texts, labels = read_IOB_file(args.input)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        for tokens, iob in tqdm(zip(texts, labels), total=len(texts)):

            text = ' '.join(tokens)
            true_ents = transform_to_dict(tokens, iob)

            # put input data and true entities
            output = {}
            output["text"] = text
            output["performers"] = true_ents.get("Artist") or []
            output["titles"] = true_ents.get("WoA") or []

            # extract with LLM
            predict_kwargs["text"] = text

            try:
                resp = llm.complete(prompt_template.format(**predict_kwargs))
                resp_json = json.loads(resp.text)
                llm_ents = [resp_json] if not "entities" in resp_json.keys() else resp_json["entities"] 
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
    parser.add_argument('-k', '--nexamples', type=int, help='Number of k few-shot examples.')
    parser.add_argument('-s', '--sampling_method', type=str, help='Sampling method for k examples. Defaults to "rand" (random).', default="rand")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()