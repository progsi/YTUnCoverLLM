import argparse
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from src.Utils import get_key
from src.Schema import EntityList
from src.Prompts import PROMPT_FEWSHOT, PROMPT_ZEROSHOT
from src.FewShot import FewShotSet
from typing import List, Union
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.program import LLMTextCompletionProgram
from pydantic_core._pydantic_core import ValidationError
from tqdm import tqdm
from src.Utils import read_IOB_file, transform_to_dict, write_jsonlines
import os 
import json

def init(model: str, few_shot_set: FewShotSet = None) -> Union[OpenAIPydanticProgram, LLMTextCompletionProgram]:
    """Load the program for structured output based on the LLM used
    Args:
        model (str): LLM name string
        few_shot_set (FewShotSet): 
    Returns:
        Union[OpenAIPydanticProgram, LLMTextCompletionProgram]: program module from Llamaindex
    """
    # set kwargs
    kwargs = {
        "output_cls": EntityList,
        "allow_multiple": False,
        "verbose": False,
    }
    if few_shot_set: 
        kwargs["prompt"] = few_shot_set.get_prompt_template()
    else:
        kwargs["prompt_template_str"] = PROMPT_ZEROSHOT

    try:
        llm = OpenAI(model=model, api_key=get_key("openai"), temperature=0.0)
        kwargs["llm"] = llm
        program = OpenAIPydanticProgram.from_defaults(**kwargs)
        print(f"{model} loaded successfully via OpenAI API.")
    except:
        llm = Ollama(model=model, temperature=0.0)
        kwargs["llm"] = llm
        program = LLMTextCompletionProgram.from_defaults(**kwargs)
        print(f"{model} loaded via Ollama.")
    return program


def main() -> None:
    args = parse_args()
    k = args.nexamples

    if k and k > 0:
        few_shot_set = FewShotSet(args.input)
        print(f"Few-Shot set successfully loaded; k={k}")
        predict_kwargs = {"k": k}
    else:
        few_shot_set = None
        predict_kwargs = {}

    program = init(args.llm, few_shot_set)
    
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
            predict_kwargs["text"] = text

            try:
                ent_list = program(**predict_kwargs)
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
    parser.add_argument('-k', '--nexamples', type=int, help='Number of k few-shot examples.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()