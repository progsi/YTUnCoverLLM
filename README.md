# In-Context-Learning for Music Entity Detection
This repo 

## Getting started
The code to run the LLMs in this repo is based on [LlamaIndex](https://docs.llamaindex.ai/en/stable/) and [Ollama](https://ollama.com/). The latter is needed for using local models. So if you like to use an OpenAI model, LlamaIndex should be sufficient. You can install the dependencies with:
```
conda env create -n env.yml;
conda activate ytuncoverllm
```
## Data

## Run Memorization Test
For these, you can use the scripts `run_memorization_test.py` (for non-pydantic LLMs like `Llama3`) or `run_memorization_test_pydantic.py` for LLMs with Pydantic (eg. `GPT-4o`).

## Generate Cloze Dataset
This is done with the script `generate_cloze.py`. 

## Run ICL-based Extraction
You need to provide the parameters for a `MODEL`, a `DATASET` and a list of `k`s.
For instance, to run a local Ollama model `mixtral` on the dataset `reddit+shsyt` with `k`s of 0,5 and 15, run:
```
./run_benchmark_json.sh mixtral reddit+shsyt 0,5,15
```
For models such as `gpt-40-mini`, you need an API key. Provide the path to a file (eg. `openai.txt`) and run: 
```
./run_ie_pydantic.sh gpt-40-mini openai.txt reddit+shsyt 0,5,15
```

## Citation
The code in this repository was used for the following papers. Please consider citing these, if you use this repository.
```
@inproceedings{hachmeier2025benchmark,
  title={A Benchmark and Robustness Study of In-Context-Learning with Large Language Models in Music Entity Detection},
  author={Hachmeier, Simon and Jäschke, Robert},
  booktitle={Proceedings of the 31th International Conference on Computational Linguistics},
  year={2025}
}
```
```
@inproceedings{hachmeier2024ie,
  title={Information Extraction of Music Entities in Conversational Music Queries},
  author={Hachmeier, Simon and Jäschke, Robert},
  booktitle={Proceedings of the 3rd Workshop on NLP for Music and Audio (NLP4MusA)},
  year={2024}
}
```
The Reddit portion of the dataset was annotated by a different work. If you cite our dataset, please also cite this work:
```
@InProceedings{Epure2023,
  title={A Human Subject Study of Named Entity Recognition (NER) in Conversational Music Recommendation Queries},
  author={Epure, Elena and Hennequin, Romain},
  booktitle={Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  month={May},
  year={2023}
}
```
