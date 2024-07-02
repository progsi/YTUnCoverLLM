import argparse
import os
import pandas as pd
from src.Wrapper import LlamaWrapper
from src.Utils import read_jsonfile, get_concat
from typing import List
import json
from src.Utils import read_IOB_file
from tqdm import tqdm


LLM_IDs = {"llama-3": "meta-llama/Meta-Llama-3-8B-Instruct"}

PROMPTS_PATH = "prompts"
TEMPERATURE = 0.000001 # can't set to 0!


def gen_json_schema(input_list: List[str]) -> dict:
    """
    Generates a JSON schema from a list of strings.
    Parameters:
        input_list (List[str]): A list of strings where each string will be a key in the JSON schema.
    Returns:
        dict: A JSON schema as a string.
    """
    if not isinstance(input_list, list):
        raise TypeError("Input should be a list of strings")
    if not all(isinstance(item, str) for item in input_list):
        raise TypeError("All items in the list should be strings")
    
    schema = {
        "type": "object",
        "properties": {},
        "required": input_list
    }
    
    for key in input_list:
        schema["properties"][key] = {"type": "string"}
    
    return schema

def write_to_jsonl(data, filename):
    """
    Writes a list of dictionaries to a jsonl file.
    
    Parameters:
    data (list): List of dictionaries to write to the file.
    filename (str): The name of the file to write to.
    """
    with open(filename, 'a') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def main():

    args = parse_args()

    # get model ID
    model = LlamaWrapper(LLM_IDs[args.model])

    # load elements of prompt
    role = read_jsonfile(os.path.join(PROMPTS_PATH, "roles.json"))[args.role]
    task = read_jsonfile(os.path.join(PROMPTS_PATH, "tasks.json"))[args.task]
    examples = read_jsonfile(os.path.join(PROMPTS_PATH, "examples.json")).get(args.examples)
    if not examples:
        examples = ""
    json_schema = read_jsonfile(os.path.join(PROMPTS_PATH, "schemas.json")).get(args.schema)
    if not examples:
        examples = ""
    prompt = role + task + examples

    texts, IOBs = read_IOB_file(args.input)

    outputs = []
    for i, text in tqdm(enumerate(texts)): 

        # NER to IOBs
        if not json_schema:
            json_schema = gen_json_schema(list(text))
            output = model.prompt_to_json(prompt, json_schema, TEMPERATURE)
        # Information extraction
        else:
            input_text = ' '.join(text)
            output = model.prompt_to_json(prompt + input_text, json_schema, TEMPERATURE)
        output["input"] = input_text
        print(output)
        outputs.append(output)
        
        # Write to jsonlines every k iterations
        if (i + 1) % args.write_every == 0:
            write_to_jsonl(outputs, args.output)
            outputs = []

    # Write any remaining outputs to the file
    if outputs:
        write_to_jsonl(outputs,  args.output)


def parse_args():
    parser = argparse.ArgumentParser(description='Run named entity recognition for song attribute extraction on an input dataset.')
    parser.add_argument('-i', '--input', type=str, help='Path with input text file with IOB dataset.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output jsonlines file.')
    parser.add_argument('-r', '--role', type=str, help='Which role for LLM. A key in the roles.json', default="pop_music")
    parser.add_argument('-t', '--task', type=str, help='Which task for LLM. A key in the tasks.json', default="ner_schema")
    parser.add_argument('-e', '--examples', type=str, help='Which few shot examples for LLM. A key in the examples.json.', default=None)
    parser.add_argument('-s', '--schema', type=str, help='Which json schema to use.', default=None)
    parser.add_argument('-m', '--model', type=str, help='Model to use.', choices=["llama-3"], default="llama-3")
    parser.add_argument('-k', '--write_every', type=int, help='Every how many iterations to write the json lines file.', default=50)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()