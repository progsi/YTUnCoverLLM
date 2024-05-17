import argparse
import os
import pandas as pd
from src.Wrapper import LlamaWrapper
from src.Utils import read_jsonfile, get_concat

LLM_IDs = {"llama-3": "meta-llama/Meta-Llama-3-8B-Instruct"}

PROMPTS_PATH = "prompts"
TEMPERATURE = 0.0001 # can't set to 0!

def main():

    args = parse_args()

    # load elements of prompt
    role = read_jsonfile(os.path.join(PROMPTS_PATH, "roles.json"))[args.role]
    task = read_jsonfile(os.path.join(PROMPTS_PATH, "tasks.json"))[args.task]
    schema = read_jsonfile(os.path.join(PROMPTS_PATH, "schemas.json"))[args.schema]
    examples = read_jsonfile(os.path.join(PROMPTS_PATH, "examples.json")).get(args.examples)

    data = pd.read_parquet(args.input)
    # get model ID
    model = LlamaWrapper(LLM_IDs[args.model])

    video_metadata = get_concat(data, attrs=["video_title", "description"])

    base_prompt = role + task + " The video title is: "
    outputs = []
    for text in video_metadata:
        prompt = base_prompt + f"'{text}'"
        if schema is not None:
            output = model.prompt_to_json(prompt, schema, TEMPERATURE)
        else:
            output = model.prompt_to_json(prompt, schema, TEMPERATURE)
        print(output)

        outputs.append(output)

    # to dataframe
    out_data = pd.DataFrame(outputs).add_suffix("_llm")
    out_data.to_parquet(args.output)
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Run named entity recognition for song attribute extraction on an input dataset.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output parquet file.')
    parser.add_argument('-r', '--role', type=str, help='Which role for LLM.', default="linguist_music")
    parser.add_argument('-t', '--task', type=str, help='Which task for LLM.', default="translate_perf")
    parser.add_argument('-s', '--schema', type=str, help='Which JSON schema for LLM output.', default=None)
    parser.add_argument('-e', '--examples', type=str, help='Which few shot examples for LLM', default=None)
    parser.add_argument('-m', '--model', type=str, help='Model to use.', choices=["llama-3"], default="llama-3")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()