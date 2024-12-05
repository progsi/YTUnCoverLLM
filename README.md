# LLM with In-Context-Learning to Detect Music Entities

## Getting started
The code to run the LLMs in this repo is based on [LlamaIndex](https://docs.llamaindex.ai/en/stable/) and [Ollama](https://ollama.com/). The latter is needed for using local models. So if you like to use an OpenAI model, LlamaIndex should be sufficient. You can install the dependencies with:
```
conda env create -n env.yml;
conda activate ytuncoverllm
```
## Data

## Extraction with LLMs
You need to provide the parameters for a `MODEL`, a `DATASET` and a list of `k`s.
For instance, to run a local Ollama model `mixtral` on the dataset `reddit+shsyt` with `k`s of 0,5 and 15, run:
```
./run_benchmark_json.sh mixtral reddit+shsyt 0,5,15
```
For models such as `gpt-40-mini`, you need an API key. Provide the path to a file (eg. `openai.txt`) and run: 
```
./run_ie_pydantic.sh gpt-40-mini openai.txt reddit+shsyt 0,5,15
```